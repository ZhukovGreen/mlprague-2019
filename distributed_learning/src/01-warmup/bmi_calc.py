from time import sleep
import tensorflow as tf



cluster = tf.train.ClusterSpec(
    {
        "parameter_server": ["localhost:2222"],
        "worker": [
            "localhost:2223",
            "localhost:2224",
        ],
    }
)
server = tf.train.Server(
    cluster, job_name="worker", task_index=1
)

with tf.device(f"/job:parameter_server/task:0"):
    mass = tf.placeholder(tf.float16)
    height = tf.placeholder(tf.float16)


server.join()


with tf.device(
        tf.train.replica_device_setter(
            cluster=cluster,
            worker_device=f"/job:worker/task:1",
        )
):
    # Keep track how many iterations have we done, incrementing global step
    global_step = (
        tf.train.get_or_create_global_step()
    )
    increment_global_step = tf.assign(
        global_step, global_step + 1
    )

    # The actual model training logic for worker.
    # Here we are just adding "a" to itself and recording it back to
    # parameter servers.
    sq_height = tf.pow(height, 2)
    bmi = tf.div(mass, sq_height)

# MonitoredTrainingSession allows hooks to define when to stop
# the distributed workloads. Here we stop all workers when
# global step number has been incremented by total of 10 times.
is_chief = task_id == 0
hooks = [tf.train.StopAtStepHook(last_step=10)]
with tf.train.MonitoredTrainingSession(
        server.target, is_chief=is_chief, hooks=hooks
) as sess:
    while not sess.should_stop():
        sleep(1)
        result = sess.run(
            [bmi, {"height": 1.80, "weight": 80}]
        )
        print(
            f"step: {result[1]}, a: {result[0]}"
        )
