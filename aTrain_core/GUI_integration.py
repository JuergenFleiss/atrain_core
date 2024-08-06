from multiprocessing import Queue
from queue import Full
from typing import List
import json
import os

class EventSender:
    def __init__(self, maxsize : int = 10):
        self.listeners : List[Queue]= []
        self.maxsize : int = maxsize

    def stream(self):
        listener = Queue(maxsize=self.maxsize)
        self.listeners.append(listener)
        while True:
            event = listener.get()
            if event == "stop":
                break
            yield event

    def end_stream(self):
        self.__send(data = None, event = None, stop = True)

    def __send(self, data, event, stop: bool = False):
        event_string = f"event: {event}\ndata: {data}\n\n" if not stop else "stop"
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(event_string)
            except Full:
                del self.listeners[i]

    def task_info(self, task : str):
        """Send the current task to the frontend for display during transcription."""
        self.__send(data = task, event="task")

    def error_info(self, error : str, traceback : str = ""):
        """Send an error message to the frontend for display during transcription."""
        error_data = json.dumps({"error" : error, "traceback" : traceback})
        self.__send(data = error_data, event="error")

    def progress_info(self, current : int, total : int):
        """Send a progress update to the frontend for diplay during transcription."""
        progress_data = json.dumps({"current" : current, "total" : total})
        self.__send(data = progress_data, event="progress")

    def finished_info(self):
        """Send an event to the frontend to inidcate that a process has finished."""
        self.__send(data="", event="finished")


class ProgressTracker:
    def __init__(self, total_chunks):
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self.progress_data = []

    def progress_callback(self):
        self.completed_chunks += 1
        overall_progress = (self.completed_chunks / self.total_chunks) * 100
        progress_info = {"current": self.completed_chunks, "total": self.total_chunks, "percentage": overall_progress}
        self.progress_data.append(progress_info)
        return progress_info

    def get_progress(self):
        return self.progress_data