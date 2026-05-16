import torch.multiprocessing as mp
import time

def f(q):
    q.put('hello')

if __name__ == '__main__':
    print("Starting test...")
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=f, args=(q,))
    p.start()
    print("Result:", q.get())
    p.join()
    print("Test finished.")
