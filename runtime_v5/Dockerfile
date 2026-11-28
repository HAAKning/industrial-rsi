# Keep the complete official Python runtime and install both runtime packages.
# All package bytes come from the local vendor directory, without network.
FROM python:3.12-alpine AS dependencies
COPY vendor/ /tmp/wheels/
RUN python3 -m pip install --no-cache-dir --no-index --no-deps --target /opt/deps \
    --find-links /tmp/wheels numpy==1.26.4 contest-harness==1.0.5

FROM python:3.12-alpine
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CONTEST_SUBMISSION=contest_submission
COPY --from=dependencies /opt/deps/ /usr/local/lib/python3.12/site-packages/
COPY --from=dependencies /opt/deps/bin/contest-harness /usr/local/bin/contest-harness
WORKDIR /app
COPY app/ /app/
COPY harness.py /app/harness.py
RUN python3 -c "import importlib.metadata as m; import numpy, contest_harness, contest_submission, unittest, ssl, ctypes; assert m.version('contest-harness') == '1.0.5'; assert m.version('numpy') == '1.26.4'" \
 && python3 -m pip check
WORKDIR /workspace
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
