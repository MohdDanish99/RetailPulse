# RetailPulse Deployment Report

**Prepared:** 19 October 2026
**Scope:** AWS-free deployment options and implementation guidance for the repository as it currently exists.

## 1. Executive summary

RetailPulse is a batch-oriented analytics project with two user-facing services: a FastAPI API and a Streamlit dashboard. Both consume files produced by the analytics pipeline. The repository includes Dockerfiles for those services and a Docker Compose file that starts them with an optional MLflow server.

**Recommended first deployment:** one Linux virtual machine (or an on-premises Linux server) running Docker Compose, with persistent storage for the repository's data and MLflow state. Run the pipeline on that host before exposing the application, then schedule or trigger future pipeline refreshes separately. Put a TLS reverse proxy in front of the API and dashboard.

This recommendation avoids AWS and matches the project's current shared-files design. A managed container platform or Kubernetes is possible, but requires more work around shared persistent storage and pipeline scheduling. This report does not assume a specific vendor or include vendor pricing.

## 2. Application and data flow

```text
Input transaction and inventory CSVs
                 |
                 v
     Sample data / ETL pipeline
                 |
                 v
      data/processed/* files
           /             \
          v               v
    FastAPI :8000     Streamlit :8501
                           |
                           v
                      Web browser

Optional experiment tracking: MLflow :5000
```

### Components

| Component | Repository implementation | Deployment implications |
|---|---|---|
| Sample data generation | `src/etl/generate_sample_data.py` | Writes a synthetic transaction CSV and inventory CSV under `data/raw`. Useful for a demo; not business data. |
| ETL | `src/etl/pipeline.py` and `src/etl/clean.py` | Reads `data/raw/online_retail_II.csv`; writes `transactions.parquet` and `daily_sales.parquet` under `data/processed`. |
| Forecasting | `src/forecasting/train.py` | Produces `forecasts.csv`; logs experiments through the MLflow Python client. |
| Segmentation | `src/segmentation/rfm.py` | Produces customer segment output under `data/processed`. |
| Churn | `src/churn/train.py` | Produces churn scores and uses MLflow tracking. |
| Inventory optimization | `src/inventory/optimizer.py` | Requires the inventory input and forecast output; writes an inventory plan. |
| Drift monitoring | `src/monitoring/drift.py` | Runs as a batch stage against processed data. |
| API | `src/api/main.py`, started by `docker/Dockerfile.api` | Reads generated CSV outputs directly from `data/processed`. Includes `/health`, `/docs`, forecasts, segments, churn, and inventory routes. |
| Dashboard | `dashboard/app.py`, started by `docker/Dockerfile.dashboard` | Reads generated Parquet/CSV outputs directly from `data/processed`; it does not currently fetch data from the API. |
| MLflow | Compose service | Uses SQLite and local artifact storage on a mounted volume. Keep it private unless access control is added. |

### Important runtime dependency

Building and starting the API and dashboard containers does not run the pipeline. The dashboard requires `daily_sales.parquet`; API analytical endpoints require their corresponding CSV files. The API `/health` route currently reports `{"status":"ok"}` without checking those files, so a successful health response alone does not prove the application has usable data.

## 3. AWS-free deployment choices

| Option | Fit for current repository | Tradeoffs |
|---|---|---|
| **Linux VM + Docker Compose (recommended)** | Strong fit for demo, internal pilot, or small installation. Both services can mount the same durable `data` directory. | You manage OS updates, Docker, TLS, backups, access controls, and availability. |
| **On-premises Linux server + Docker Compose** | Same deployment shape when the service must stay on a private network or existing hardware. | Your organization also manages hardware capacity, network reachability, and recovery. |
| **Managed container hosting** | Can run API and dashboard as separate containers. | Confirm durable/shared storage support. Add a batch job or scheduler and ensure all services see the generated data. Vendor-specific setup varies. |
| **Non-AWS Kubernetes** | Appropriate if Kubernetes is already operated by the team. The repository has a starting manifest. | The manifest is incomplete for a production rollout: no ingress/TLS, pipeline workload, MLflow workload, or image publishing workflow. Shared-volume support and access policies must be designed. |
| **Vercel** | The root `app.py` exposes a FastAPI app, making an API-focused experiment conceivable. | It does not deploy the Streamlit dashboard or batch pipeline as a complete application, and its execution/storage model does not solve the generated-file dependency. Not recommended as the full deployment. |

## 4. Recommended architecture

```text
                         +--------------------------+
Browser -- HTTPS ------>| Reverse proxy (Caddy or  |
                         | Nginx; TLS and routing)  |
                         +------------+-------------+
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
                 Streamlit :8501             FastAPI :8000
                         \                          /
                          \                        /
                           +----------------------+
                           | Persistent host data |
                           | data/raw             |
                           | data/processed       |
                           +----------------------+

Pipeline run on host or controlled job --> data/processed
MLflow :5000 (optional; private access only)
```

The existing Compose configuration maps host port 8000 to the API, 8501 to the dashboard, and 5000 to MLflow. On an internet-facing server, publish only the reverse proxy's HTTP/HTTPS ports to the internet. Bind internal service ports to localhost or place them on an internal Docker network. In particular, do not expose MLflow directly without authentication and a reason to make it reachable.

The current dashboard is a standalone Streamlit interface. A path-prefix arrangement such as `/dashboard` may require proxy and Streamlit base-path settings. The simplest first setup is to use separate hostnames (for example, a dashboard hostname and an API hostname) or otherwise verify the selected proxy configuration with Streamlit's websocket traffic. Keep API documentation at `/docs` private or protected if it reveals endpoints that should not be public.

## 5. Step-by-step deployment on a Linux VM

The commands below assume a Debian/Ubuntu-like Linux host, Docker Engine with the Compose plugin, Git, Python 3.11, and GNU Make. Adjust package installation to the chosen operating system. Avoid using this procedure as a substitute for an organization's data security and access policies.

### 5.1 Prepare the host and checkout

1. Create a Linux VM with enough memory and disk for the input dataset, processed files, model artifacts, logs, and backups. Model training can use more resources than the serving containers; size the host based on actual data and workload.
2. Install Docker Engine, the Docker Compose plugin, Git, Python, and `make` using the operating system's supported package instructions.
3. Check out the project into a stable directory. Keep persistent application data outside ephemeral container layers; the existing Compose bind mount uses the checkout's `data` directory.
4. Configure a firewall to allow SSH only from trusted administration locations and web traffic only on ports 80 and 443. Do not open ports 8000, 8501, or 5000 to the public internet.

### 5.2 Prepare data and generate outputs

For a sample-data demonstration, from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
make all
```

`make all` runs these stages in order: sample-data generation, ETL, forecasting, segmentation, churn, inventory optimization, and drift monitoring. The generated sample data is synthetic and should not be presented as real business results.

For real data, place the transaction file at `data/raw/online_retail_II.csv` and the inventory file at `data/raw/inventory.csv`, matching the column formats expected by the code. The transaction dataset is expected to have fields such as `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, and `Country`. Inventory optimization expects `StockCode`, `stock_on_hand`, `lead_time_days`, and `unit_cost`. Validate the input with a representative non-production dataset before using the pipeline operationally.

The `make all` target always runs the sample-data generator first, which writes those sample input paths. Therefore, do **not** run `make all` unchanged when you intend to process real input: it will replace the input files with generated data. For real data, run the stages intentionally, beginning with `make etl`, then the needed model/analytics stages in dependency order (for example, forecast before inventory). Verify the actual outputs before starting or refreshing the serving services. Consider changing the Makefile to provide separate `demo` and `process` targets before making this a routine production operation.

### 5.3 Build and start serving containers

After the required processed files have been generated:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=100 api dashboard
```

For a private smoke check on the VM:

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/
```

Then check an analytical API route using a SKU or customer ID that exists in the generated data, and open the dashboard through the configured proxy. `/health` only confirms the process is responding; the root endpoint and data-dependent routes help confirm application data is available.

The Compose file currently mounts `./data:/app/data` into both service containers. Preserve the `data` directory across container recreation and code releases. The dashboard container also has `src` copied into it so it can import the shared path configuration.

### 5.4 Configure HTTPS and external access

Install and configure a reverse proxy on the VM or as a separately managed service. Use a valid domain and TLS certificates. Route requests to the internal API and dashboard endpoints; configure WebSocket forwarding for Streamlit. Ensure the public firewall exposes only ports 80/443 (and restricted administrative SSH). Restrict MLflow and API documentation to trusted users if they are not intended for public access.

The repository does not contain a reverse-proxy configuration, domain setup, authentication layer, or certificate provisioning. These must be added for the selected host and deployment policy.

### 5.5 Updates and rollback

For a release, update the checkout to a reviewed version, rebuild the images, and recreate the services:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=100 api dashboard
```

Keep a known-good image or source revision and a backup of the data before updating. Do not delete or overwrite `data/processed` during an update unless you intend to regenerate it. If outputs are regenerated, validate them before directing users to the new results. A rollback should restore both a compatible application revision and the matching data/artifacts when formats change.

## 6. Pipeline and refresh design

There is no scheduled production pipeline in Compose. `make all` is a local convenience target; it is not a service, queue, or scheduler. Choose an operating model:

- **Manual refresh:** an operator uploads validated inputs and runs the documented stages over SSH or through a controlled administrative process.
- **Scheduled refresh:** use the host's scheduler or an external job runner to execute a dedicated script at a defined cadence. Capture logs and exit codes, avoid overlapping runs, and alert on failure.
- **Future job service:** package the pipeline as a separate container/job with a pinned environment and explicit mounted input/output volumes.

Before automating, split demo sample generation from real-data processing, define a validation step, and make output publication safe. A failed pipeline should not leave users reading a mixture of old and partially refreshed files. One improvement is to write a complete output set to a versioned staging directory, validate it, then switch the active output set atomically.

## 7. Persistence, backups, and security

### Persistence

- The Compose `./data:/app/data` bind mounts make host files durable across container recreation, but only as durable as the VM disk and backup policy.
- MLflow uses `./mlflow-data:/mlflow` for its database and artifacts. Back this up if experiment history matters.
- The MLflow service command refers to `/mlflow/mlflow.db`; the checked-in `mlflow.db` at repository root is not automatically the same mounted database path.
- Training code calls the MLflow Python API but does not visibly configure a tracking URI to the Compose MLflow server. Verify the tracking setup before assuming model runs are stored in that server; by default the client may write to a local tracking directory.

### Security and operations

- Restrict host access and container ports; add authentication/authorization appropriate for the users and data.
- Keep credentials out of source control and image layers. Use host-managed secrets or a secret store supported by the deployment platform.
- Protect customer-level information in logs, backups, and the dashboard. Define retention and restore access.
- Add service restart policies, meaningful health checks, resource limits appropriate to measured use, log rotation, host monitoring, and disk-space alerts.
- Test backups by restoring them to a separate environment. A backup that has never been restored is not a proven recovery process.
- Decide how to handle stale data during pipeline failures and display the last successful refresh time to users.

## 8. Kubernetes notes

`k8s/retailpulse.yaml` defines a PVC, API and dashboard deployments, and services. It is useful as a starting point, but it does not by itself deploy a complete public application:

- Images are named `retailpulse-api:latest` and `retailpulse-dashboard:latest`, which must be built and made available to the cluster's nodes or pushed to a registry.
- The PVC must be provisioned by a compatible storage class and populated with the expected data. A single `ReadWriteOnce` claim may not be mountable by API and dashboard pods on separate nodes, depending on the storage provider. Validate the access mode and scheduling behavior.
- No pipeline job, MLflow deployment, ingress, TLS, authentication, secrets, or backup policy is included.
- The dashboard service is exposed as a `NodePort`; decide whether that is acceptable for the environment or replace it with controlled ingress.
- Replicas are set to one. Increasing replicas requires shared data semantics and appropriate storage behavior.

For these reasons, use the Compose VM route for the first installation unless the team already operates Kubernetes and can fill in these platform pieces.

## 9. Deployment blockers and improvements

### Resolve before release builds

1. **Align Python versions.** `pyproject.toml` requires Python `>=3.12`, while the Dockerfiles and CI workflow use Python 3.11. Select a supported version and align project metadata, CI, training, and serving environments.
2. **Separate demo inputs from production inputs.** `make all` starts with synthetic data generation and overwrites the raw input paths. Create distinct targets and document the real-data processing sequence.
3. **Make dependencies reproducible.** Requirements use version lower bounds rather than a complete lock. Pin or lock the build/runtime environments and validate the complete pipeline and serving images against them.
4. **Define pipeline execution and failure behavior.** The current Compose deployment does not run or schedule analytics.
5. **Configure the MLflow client deliberately.** Set the tracking URI and artifact location for the intended tracking server, then verify a training run is recorded there.

### Resolve before public or business use

- Add TLS, access controls, secret handling, firewall policy, and a private strategy for MLflow.
- Add health checks that verify required data readiness, as well as service restart behavior and logs/metrics.
- Define a data refresh cadence, output validation, rollback, backup, retention, and restore procedure.
- Decide how user access to customer data and API endpoints is authenticated and audited.
- Add a release process that publishes immutable, versioned images rather than relying on mutable `latest` tags.

## 10. Suggested rollout stages

### Stage 1 — Demonstration

Use one VM, synthetic sample data, persistent disk, Compose, and HTTPS. Keep MLflow private or omit it if experiment tracking is unnecessary. Clearly label the data and results as synthetic.

### Stage 2 — Internal pilot

Use representative, approved real data. Establish a controlled manual refresh, access restrictions, backups, monitoring, pinned dependencies, and a restore test. Confirm that dashboard and API outputs correspond to the same successful pipeline run.

### Stage 3 — Regular operation and growth

Automate validated pipeline runs with failure alerts and versioned outputs. If data volume or availability needs outgrow one host, move generated data to a shared database/object storage solution offered by the chosen non-AWS platform, and deploy pipeline and serving components independently. Add redundancy and an explicit recovery objective before increasing replicas.

## 11. Deployment checklist

- [ ] Linux host provisioned, patched, and reachable only through approved administration paths.
- [ ] Docker Engine and Compose installed; selected Python version is aligned across project configuration and CI.
- [ ] Domain and HTTPS reverse proxy configured; only intended ports are public.
- [ ] Real input format validated, or demo data clearly identified as synthetic.
- [ ] Pipeline run completed successfully and required files exist under `data/processed`.
- [ ] API root and data-dependent API endpoint checked; dashboard opened through HTTPS.
- [ ] Persistent data and MLflow state paths documented and backed up.
- [ ] MLflow access and tracking URI verified, or MLflow disabled.
- [ ] Refresh procedure, failure behavior, logs, disk alerts, and restore steps documented.
- [ ] Release image versions recorded and rollback procedure understood.

## 12. Conclusion

RetailPulse can be deployed without AWS. The current implementation is best served initially by a single Linux VM running Docker Compose, with persistent host data, a separately managed pipeline run, and HTTPS routing through a reverse proxy. The critical operational detail is that both web services read generated files and Compose does not create or refresh those files. Before using real data, separate sample generation from the real-data workflow, align the Python versions, and establish access control, backups, and a repeatable pipeline process.

**Repository areas reviewed:** README, project metadata, requirements, Makefile, Compose file, Dockerfiles, API, dashboard, ETL and analytics modules, CI workflow, Kubernetes manifest, and Vercel entry point.
