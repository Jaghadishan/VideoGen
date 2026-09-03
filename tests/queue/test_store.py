import threading
from datetime import datetime

from app.queue import store
from app.queue.models import Job, JobStatus

from tests.conftest import make_brief


def test_next_job_id_format_and_increment():
    first = store.next_job_id()
    second = store.next_job_id()

    date_part, time_part, counter_part = first.split("_")
    assert len(date_part) == 8
    assert len(time_part) == 6
    assert counter_part == "000001"
    assert second.endswith("_000002")


def test_next_job_id_has_no_collisions_under_concurrency():
    ids: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        job_id = store.next_job_id()
        with lock:
            ids.append(job_id)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == len(set(ids))


def test_save_and_load_job_round_trip():
    job_id = store.next_job_id()
    job = Job(job_id=job_id, created_at=datetime.now(), brief=make_brief())
    store.save_job(job)

    loaded = store.load_job(job_id)
    assert loaded.job_id == job_id
    assert loaded.status == JobStatus.PENDING
    assert loaded.brief.title == job.brief.title


def test_list_jobs_sorts_by_counter_not_lexicographic_job_id():
    # "01012027..." sorts before "19082026..." as a plain string, but the
    # counter says the second job was created first — list_jobs() must
    # trust the counter, not the date-prefixed job_id string.
    first_created = Job(job_id="19082026_143207_000001", created_at=datetime.now(), brief=make_brief())
    second_created = Job(job_id="01012027_090000_000002", created_at=datetime.now(), brief=make_brief())

    store.save_job(second_created)
    store.save_job(first_created)

    listed = store.list_jobs()
    assert [job.job_id for job in listed] == [first_created.job_id, second_created.job_id]
