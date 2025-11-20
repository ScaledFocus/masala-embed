# Benchmarking

## Running Locust

```
locust -f locustfile.py --headless --host http://127.0.0.1:8000 -u 50 -r 10 -t 1m --csv=run --loglevel DEBUG
```

### Parameters:

1. `-f`: Path to the Locust file to run.
2. `-u`: Number of users to simulate.
3. `-r`: Rate at which users are spawned.
4. `-t`: Duration of the test.
5. `--csv`: Name of the CSV file to save the results.

### Environment Variables:

1. `HOST`: The target URL to benchmark.
