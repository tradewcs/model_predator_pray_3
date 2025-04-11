from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

def simulate(params):
    tau1 = params['tau1']
    tau2 = params['tau2']
    tau = params['tau']

    a1 = params['a1']
    r1 = params['r1'] / (1 + a1)
    K1 = params['K1']

    a2 = params['a2']
    r2 = params['r2'] / (1 + a2)
    K2 = params['K2']

    b1 = params['b1']
    b2 = 1 - b1
    r = params['r']
    K = params['K']

    N1_0 = params['N1_0']
    N2_0 = params['N2_0']
    N_0 = params['N_0']

    t0 = 0
    tf = 40
    n = 500

    def model(Y, t):
        N1_t, N2_t, N_t = Y(t)
        N1_tau = Y(t - tau1)[0]
        N2_tau = Y(t - tau2)[1]
        N_tau = Y(t - tau)[2]

        dN_1_dt = r1 * (1 + a1 * (1 - N_t / K) - N1_tau / K1) * N1_t
        dN_2_dt = r2 * (1 + a2 * (1 - N_t / K) - N2_tau / K2) * N2_t
        dN_t_dt = r * (b1 * N1_t / K1 + b2 * N2_t / K2 - N_tau / K) * N_t

        return np.array([dN_1_dt, dN_2_dt, dN_t_dt])

    def initial_history(t):
        return np.array([N1_0, N2_0, N_0])

    tt = np.linspace(t0, tf, n)
    solution = ddeint(model, initial_history, tt)

    plt.figure()
    plt.plot(tt, solution[:, 0], label='N1(t)')
    plt.plot(tt, solution[:, 1], label='N2(t)')
    plt.plot(tt, solution[:, 2], label='N(t)')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')

    filename = 'static/plot.png'
    plt.savefig(filename)
    plt.close()

    return filename

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/simulate", response_class=HTMLResponse)
async def simulate_form(request: Request,
                        tau1: float = Form(...),
                        tau2: float = Form(...),
                        tau: float = Form(...),
                        a1: float = Form(...),
                        r1: float = Form(...),
                        K1: float = Form(...),
                        a2: float = Form(...),
                        r2: float = Form(...),
                        K2: float = Form(...),
                        b1: float = Form(...),
                        r: float = Form(...),
                        K: float = Form(...),
                        N1_0: float = Form(...),
                        N2_0: float = Form(...),
                        N_0: float = Form(...)):
    params = {
        "tau1": tau1,
        "tau2": tau2,
        "tau": tau,
        "a1": a1,
        "r1": r1,
        "K1": K1,
        "a2": a2,
        "r2": r2,
        "K2": K2,
        "b1": b1,
        "r": r,
        "K": K,
        "N1_0": N1_0,
        "N2_0": N2_0,
        "N_0": N_0
    }
    simulate(params)
    return templates.TemplateResponse("index.html", {"request": request, "plot_url": "/static/plot.png"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
