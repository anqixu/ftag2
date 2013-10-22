#ifndef RUNNABLE_H
#define RUNNABLE_H


#include <atomic>
#include <thread>


class Runnable {
public:
  Runnable() : m_stop(false), m_thread() {
  };

  virtual ~Runnable() {
    try {
      stop();
    } catch(...) {
      /*??*/
    }
  };

  Runnable(Runnable const&) = delete;
  Runnable& operator=(Runnable const&) = delete;

  /** NOTE: ideally need to put in try-catch clause */
  void stop() {
    m_stop = true;
    if (m_thread.joinable()) m_thread.join();
  };
  inline void announceStop() { m_stop = true; };

  void start() {
    m_thread = std::thread(&Runnable::run, this);
  };
  inline bool isRunning() { return !m_stop; };

protected:
  virtual void run() = 0;

  std::atomic<bool> m_stop;
  std::thread m_thread;
};


#endif // RUNNABLE_H
