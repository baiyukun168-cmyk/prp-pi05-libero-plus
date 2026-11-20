import json
import logging
from pathlib import Path
import sys
import time

from openpi.policies import policy_config
from openpi.training import config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO)
output = Path(sys.argv[1])
start = time.perf_counter()
policy = policy_config.create_trained_policy(
    config.get_config("pi05_libero"), "/root/prp/checkpoints/pi05_libero"
)
(output / "model-load.json").write_text(json.dumps({
    "load_s": time.perf_counter() - start,
    "config": "pi05_libero", "sample_kwargs": {},
    "checkpoint": "/root/prp/checkpoints/pi05_libero",
}, indent=2))
WebsocketPolicyServer(policy, host="CLOUD_HOST", port=8011, metadata=policy.metadata).serve_forever()
