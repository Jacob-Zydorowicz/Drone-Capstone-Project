//
// Created by connputer on 10/28/23.
//

#ifndef DATACOLLECTION_THREADPOOL_H
#define DATACOLLECTION_THREADPOOL_H

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool {
public:


    ThreadPool(size_t thread_count) : stop(false)
    {
        for (size_t i = 0; i < thread_count; ++i)
        {
            pool.emplace_back([this]
            {
                while (true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : pool) {
            worker.join();
        }
    }

    template <typename Func>
    void enqueue(Func func) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::move(func));
        }
        condition.notify_one();
    }

private:
    std::vector<std::thread> pool;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};


#endif //DATACOLLECTION_THREADPOOL_H
