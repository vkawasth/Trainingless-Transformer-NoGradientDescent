#pragma once
// task_queue.hpp
// A worker-thread task queue, evolved from the single-threaded, move-only,
// future-returning TaskQueue in consteval_constexpr.cpp.
//
// Key differences from the original:
//   - A dedicated worker thread drains the queue (original ran inline).
//   - submit() is thread-safe and wakes the worker via condition_variable.
//   - Each task is a std::function<void()> that fulfils its own promise, so we
//     keep the "return a std::future" contract without templating the queue on
//     a single signature -- different ops can return different types.
//
// IMPORTANT (GIL): this queue knows NOTHING about Python. It just runs C++
// callables on one worker thread. The Python/GIL discipline lives entirely
// inside the task bodies submitted from the bridge (see py_bridge.*). Keeping
// the queue Python-agnostic is deliberate: it stays reusable and testable
// without an interpreter.

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>
#include <utility>
#include <atomic>
#include <stdexcept>

namespace bridge {

class TaskQueue {
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stopping_{false};
    std::thread worker_;

    void run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lk(mtx_);
                cv_.wait(lk, [&] { return stopping_.load() || !tasks_.empty(); });
                if (stopping_.load() && tasks_.empty()) return;
                job = std::move(tasks_.front());
                tasks_.pop();
            }
            job();  // runs outside the lock so submit() never blocks on a task
        }
    }

public:
    TaskQueue() : worker_([this] { run(); }) {}

    ~TaskQueue() { shutdown(); }

    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    // Submit any callable. Returns std::future<result-of-callable>.
    // The callable is invoked on the worker thread. Exceptions thrown inside
    // it are captured into the future (standard promise/future semantics).
    template <class F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using R = std::invoke_result_t<std::decay_t<F>>;

        // package_task gives us the promise/future plumbing for free and
        // forwards exceptions correctly. It's move-only, so we wrap it in a
        // shared_ptr to store inside std::function (which needs copyable).
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard lk(mtx_);
            if (stopping_.load())
                throw std::runtime_error("submit after shutdown");
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    void shutdown() {
        bool expected = false;
        if (stopping_.compare_exchange_strong(expected, true)) {
            cv_.notify_all();
            if (worker_.joinable()) worker_.join();
        }
    }
};

} // namespace bridge
