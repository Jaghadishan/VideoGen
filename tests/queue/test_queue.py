from app.queue import queue, store
from app.queue.models import JobStatus

from tests.conftest import make_brief


def test_submit_creates_pending_job():
    job = queue.submit(make_brief())

    assert job.status == JobStatus.PENDING
    assert job.needs_script_draft is False
    assert store.load_job(job.job_id).job_id == job.job_id


def test_submit_carries_needs_script_draft():
    job = queue.submit(make_brief(), needs_script_draft=True)
    assert job.needs_script_draft is True


def test_pending_jobs_returns_fifo_order():
    first = queue.submit(make_brief())
    second = queue.submit(make_brief())

    assert [job.job_id for job in queue.pending_jobs()] == [first.job_id, second.job_id]


def test_queue_position_reflects_fifo_order():
    first = queue.submit(make_brief())
    second = queue.submit(make_brief())

    assert queue.queue_position(first.job_id) == 1
    assert queue.queue_position(second.job_id) == 2


def test_queue_position_is_none_once_not_pending():
    job = queue.submit(make_brief())
    job.status = JobStatus.DONE
    store.save_job(job)

    assert queue.queue_position(job.job_id) is None


def test_next_pending_returns_earliest_job():
    first = queue.submit(make_brief())
    queue.submit(make_brief())

    assert queue.next_pending().job_id == first.job_id


def test_next_pending_is_none_when_queue_empty():
    assert queue.next_pending() is None


def test_active_job_detects_in_progress_status():
    job = queue.submit(make_brief())
    assert queue.active_job() is None

    job.status = JobStatus.GENERATING_VIDEO
    store.save_job(job)

    assert queue.active_job().job_id == job.job_id
