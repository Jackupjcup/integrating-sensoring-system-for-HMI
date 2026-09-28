from datetime import datetime

def init_datetime():
    d_t = str(datetime.now())
    d_t = d_t.replace(' ', '_')
    d_t = d_t.replace(':', '_')

    return d_t[: 19]