# Science CPU sandbox image

Scientific Python tools for the agent sandbox, based on the validated Ubuntu 22.04 ml-base image. Includes symbolic and numerical computation, PDF extraction, dataframes and datasets, chemistry, and optimization solvers.

Build (Docker data root on this workspace is `/data/docker`):

```bash
docker build --progress=plain \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  --build-arg http_proxy --build-arg https_proxy \
  --build-arg NO_PROXY --build-arg no_proxy \
  -t science-agent-cpu:20261010 images/science-cpu
```

The build uses official PyPI by default and disables the base image's pip configuration for installation. Override `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL` with build arguments if needed. Docker's predefined proxy build arguments do not become runtime environment variables.

Run the smoke checks with the same CPU and memory limits as the tool sandbox:

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -v "$PWD/images/science-cpu/smoke.py:/tmp/science_cpu_verify.py:ro" \
  --entrypoint python3 science-agent-cpu:20261010 /tmp/science_cpu_verify.py
```

The checks run calculations, PDF extraction, Parquet round trips, local dataset operations, chemistry parsing, several optimization solvers, and `pip check` without network access.

SageMath/PARI and OCR engines are not included. PDF text extraction is provided by PyMuPDF, pypdf/PyPDF2, pdfplumber, pikepdf, and the Poppler command-line tools; importing the optional `pdftotext` Python wrapper is not supported.
