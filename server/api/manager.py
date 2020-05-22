import queue
import threading
import warnings
import traceback


class Payload():
    def __init__(self, task, data):
        self.cv = threading.Condition()
        self.task = task
        self.data = data
        self.result = None
        self.rejected = False


class ModelManager():
    def __init__(self, n_workers=2, queue_capacity=100, send_timeout=5.0, \
            processing_timeout=20.0, worker_waiting_timeout=1.0):
        self.queue = queue.Queue(queue_capacity)
        self.shutdown = False
        self.workers = [threading.Thread(target=self._worker) for _ in range(n_workers)]

        self.send_timeout = send_timeout
        self.processing_timeout = processing_timeout
        self.worker_waiting_timeout = worker_waiting_timeout


    def __enter__(self):
        for worker in self.workers:
            worker.start()

        return self


    def __exit__(self, type, value, traceback):
        print("Server is shutting down")
        for _ in range(len(self.workers)):
            self.queue.put(None)

        self.shutdown = True
        for worker in self.workers:
            worker.join()


    def send_task(self, payload):
        if self.shutdown:
            raise RuntimeError("Server is shutting down")

        with payload.cv:
            # throws queue.Full
            self.queue.put(payload, timeout=self.send_timeout)
            
            if not payload.cv.wait(timeout=self.processing_timeout):
                payload.rejected = True
                raise RuntimeError("Request rejected by timeout")
        
        if payload.rejected:
            raise RuntimeError("Request was rejected")

        return payload.result

    
    def _worker(self):
        while True:
            try:
                payload = self.queue.get(timeout=self.worker_waiting_timeout)
            except queue.Empty:
                continue
            # Shutdown by None element in queue
            if payload is None:
                break
            else:
                with payload.cv:
                    if payload.rejected:
                        warnings.warn("Request was rejected by requester", RuntimeWarning)
                        self.queue.task_done()
                        continue

                    if self.shutdown:
                        payload.rejected = True
                    else:
                        try:
                            payload.result = payload.task(*payload.data)
                        except Exception as e:
                            warnings.warn("Exception occurred during evaluation: {}".format(e), RuntimeWarning)
                            traceback.print_exc()
                            payload.rejected = True
                    
                    payload.cv.notify()
                self.queue.task_done()
