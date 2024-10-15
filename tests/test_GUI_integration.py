import unittest
import json
import threading
import time
from queue import Queue, Empty
from aTrain_core.GUI_integration import (
    EventSender,
    ProgressTracker,
)  # Adjust import as necessary


class TestEventSender(unittest.TestCase):
    def setUp(self):
        self.event_sender = EventSender(maxsize=5)

    def test_event_sending(self):
        listener = Queue(maxsize=5)
        self.event_sender.listeners.append(listener)

        self.event_sender.task_info("Transcribing audio")

        event = listener.get()
        self.assertIn("event: task", event)
        self.assertIn("data: Transcribing audio", event)

    def test_error_sending(self):
        listener = Queue(maxsize=5)
        self.event_sender.listeners.append(listener)

        self.event_sender.error_info("An error occurred", "Traceback details")

        event = listener.get()
        error_data = json.loads(event.split("data: ")[1])
        self.assertEqual(error_data["error"], "An error occurred")
        self.assertEqual(error_data["traceback"], "Traceback details")

    def test_progress_sending(self):
        listener = Queue(maxsize=5)
        self.event_sender.listeners.append(listener)

        self.event_sender.progress_info(5, 10)

        event = listener.get()
        progress_data = json.loads(event.split("data: ")[1])
        self.assertEqual(progress_data["current"], 5)
        self.assertEqual(progress_data["total"], 10)

    def test_finished_event(self):
        listener = Queue(maxsize=5)
        self.event_sender.listeners.append(listener)

        self.event_sender.finished_info()

        event = listener.get()
        self.assertEqual(event, "event: finished\ndata: \n\n")

    def test_streaming_events(self):
        listener = Queue(maxsize=5)
        self.event_sender.listeners.append(listener)

        # Start streaming in a separate thread
        stream_thread = threading.Thread(target=self.event_sender.stream)
        stream_thread.daemon = (
            True  # This makes sure the thread exits when the main program exits
        )
        stream_thread.start()

        # Send a task event
        self.event_sender.task_info("Transcribing audio")

        # Attempt to get the event from the listener
        try:
            # Use a timeout to avoid blocking indefinitely
            event = listener.get(timeout=1)
            self.assertIn("event: task", event)
        except Empty:
            self.fail("No event received within timeout period")

        self.event_sender.end_stream()

        # Wait for the stream thread to process the stop event
        time.sleep(0.1)

        # Verify that the next call to get the event returns "stop"
        try:
            event = listener.get(timeout=1)
            self.assertEqual(event, "stop")
        except Empty:
            self.fail("Stop event not received within timeout period")


class TestProgressTracker(unittest.TestCase):
    def setUp(self):
        self.total_chunks = 10
        self.progress_tracker = ProgressTracker(total_chunks=self.total_chunks)

    def test_progress_callback(self):
        for i in range(5):
            progress = self.progress_tracker.progress_callback(i)
            self.assertEqual(progress["current"], i + 1)
            self.assertEqual(progress["total"], self.total_chunks)
            self.assertAlmostEqual(
                progress["percentage"], (i + 1) / self.total_chunks * 100
            )

    def test_get_progress(self):
        for i in range(5):
            self.progress_tracker.progress_callback(i)

        progress_data = self.progress_tracker.get_progress()
        self.assertEqual(len(progress_data), 5)  # Ensure we have 5 progress entries
        for i, progress in enumerate(progress_data):
            self.assertEqual(progress["current"], i + 1)
            self.assertEqual(progress["total"], self.total_chunks)


if __name__ == "__main__":
    unittest.main()
