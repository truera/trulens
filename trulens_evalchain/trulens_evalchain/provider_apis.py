from multiprocessing.pool import ThreadPool
from queue import Queue
from time import sleep
from typing import Any, Dict, Hashable, Sequence

import requests
from tqdm.auto import tqdm


class SingletonPerName():
    """
    Class for creating singleton instances except there being one instance max,
    there is one max per different `name` argument. If `name` is never given,
    reverts to normal singleton behaviour.
    """

    # Hold singleton instances here.
    instances: Dict[Hashable, 'SingletonPerName'] = dict()

    def __new__(cls, name: str = None, *args, **kwargs):
        """
        Create the singleton instance if it doesn't already exist and return it.
        """

        if name not in cls.instances:
            print(f"creating {cls} instance with name = {name}")
            cls.instances[name] = super().__new__(cls)

        return cls.instances[name]


class TP(SingletonPerName):  # "thread processing"

    def __init__(self):
        self.thread_pool = ThreadPool(processes=8)


class Endpoint(SingletonPerName):

    def __init__(
        self, name: str, rpm: float = 60, retries: int = 3, post_headers=None
    ):
        """
        
        Args:
        - name: str -- api name / identifier.
        - rpm: float -- requests per minute.
        - retries: int -- number of retries before failure.
        """

        self.rpm = rpm
        self.retries = retries
        self.pace = Queue(
            maxsize=rpm / 6.0
        )  # 10 second's worth of accumulated api
        self.tqdm = tqdm(desc=f"{name}", unit="request")
        self.name = name
        self.post_headers = post_headers

        self._start_pacer()

    def pace_me(self):
        """
        Block until we can make a request to this endpoint.
        """

        self.pace.get()
        self.tqdm.update(1)
        return

    def post(self, url, payload) -> Any:
        extra = dict()
        if self.post_headers is not None:
            extra['headers'] = self.post_headers

        self.pace_me()
        ret = requests.post(url, json=payload, **extra)

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            print(f"WARNING: Waiting for {j} ({wait_time}) second(s).")
            sleep(wait_time)
            return self.post(url, payload)

        assert isinstance(
            j, Sequence
        ) and len(j) > 0, f"Post did not return a sequence: {j}"

        return j[0]

    def run_me(self, thunk):
        """
        Run the given thunk, returning itse output, on pace with the api.
        Retries request multiple times if self.retries > 0.
        """

        retries = self.retries + 1
        retry_delay = 2.0

        while retries > 0:
            try:
                self.pace_me()
                ret = thunk()
                return ret
            except Exception as e:
                retries -= 1
                print(
                    f"WARNING: {self.name} request failed {type(e)}={e}. Retries={retries}."
                )
                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(
            f"API {self.name} request failed {self.retries+1} time(s)."
        )

    def _start_pacer(self):

        def keep_pace():
            while True:
                sleep(60.0 / self.rpm)
                self.pace.put(True)

        TP().thread_pool.apply_async(keep_pace)