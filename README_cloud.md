# Cloud run instructions

## Local sanity check (native Python)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python master_eval.py --name local_smoke --time 180 --seed 42
```

## Docker build

```bash
docker build -t thesis-master-eval:latest .
```

## Docker run (long CP-SAT verification)

Results are written inside the container under `/app/results`. Mount a host directory to persist them.

```bash
docker run --rm \
  -e RUN_NAME=cpsat_long_3600 \
  -e TIME_SEC=3600 \
  -e SEED=42 \
  -v "$(pwd)/results:/app/results" \
  thesis-master-eval:latest /app/cloud_entrypoint.sh
```

To isolate the CP-SAT bug (skip the RL methods):

```bash
docker run --rm \
  -e RUN_NAME=cpsat_only_7200 \
  -e TIME_SEC=7200 \
  -e SKIP="ql ppo rlhh" \
  -v "$(pwd)/results:/app/results" \
  thesis-master-eval:latest /app/cloud_entrypoint.sh
```

## GCP Cloud Run Jobs (one-off batch)

Cloud Run Jobs is the right GCP primitive here — this is a long batch job, not a web service.

```bash
REGION=us-central1
PROJECT=<your-gcp-project>
REPO=thesis
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/master-eval:latest"

gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION || true
gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker build -t $IMAGE .
docker push $IMAGE

gcloud run jobs create master-eval \
  --image=$IMAGE \
  --region=$REGION \
  --cpu=8 \
  --memory=16Gi \
  --task-timeout=14400s \
  --max-retries=0 \
  --command=/app/cloud_entrypoint.sh \
  --set-env-vars=RUN_NAME=cpsat_long_3600,TIME_SEC=3600,SEED=42

gcloud run jobs execute master-eval --region=$REGION --wait
```

Cloud Run Jobs has a **24h task timeout cap**. For anything longer, use Compute Engine or GKE.

## AWS / plain VM

Any 8-vCPU / 16 GB VM (e.g. AWS `c6i.2xlarge`, GCP `e2-standard-8`) runs the Docker image directly:

```bash
docker run -d --name thesis \
  -e RUN_NAME=cpsat_long_7200 \
  -e TIME_SEC=7200 \
  -v /data/results:/app/results \
  thesis-master-eval:latest /app/cloud_entrypoint.sh

docker logs -f thesis
```

## GPU variant (optional — only helps PPO/RL-HH)

Replace the torch line in `requirements.txt` with the CUDA 11.8 wheel:

```
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.7.1+cu118
```

and use an NVIDIA base image (`nvidia/cuda:11.8.0-runtime-ubuntu22.04`) instead of `python:3.11-slim`, then `docker run --gpus all ...`. CP-SAT is CPU-only; the gap issue you're chasing is a CPU-time problem, not a GPU one.

## Where to look after the run

- `results/<stamp>_<run_name>/Master_Evaluation_*.md` — headline table.
- `results/<stamp>_<run_name>/cpsat_raw_result.json` — `oracle_profit`, `best_bound`, `gap_pct`, `oracle_status`, `cpsat_schedule` vs `schedule` (replay drops).
- The CP-SAT budget needs to be large enough for `oracle_status` to read `Optimal` or `Feasible(gap<=1%)`; anything else means the solver was still improving when the clock expired.
