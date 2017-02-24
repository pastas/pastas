"""
This file contains the GUI made for pasta.

"""
try:
    # for Python2
    import Tkinter as tk
except ImportError:
    # for Python3
    import tkinter as tk

try:
    # for Python2
    import ttk
except ImportError:
    # for Python3
    from tkinter import ttk

try:
    # for Python2
    import tkMessageBox
except ImportError:
    # for Python3
    from tkinter import tkMessageBox


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pastas import *
import pickle
import os.path


class PastaModelGui(tk.Frame):
    def __init__(self, ml, parent=None):
        if parent is None:
            parent = tk.Tk()
            parent.geometry(self.center(parent, 600, 500))
        tk.Frame.__init__(self, parent)

        self.settingsFile = 'settings'
        self.load_settings()

        self.ml = ml

        self.parent = parent
        self.initUI()

        parent.mainloop()

    def initUI(self):
        # Initialize the GUI Application and create all the frames

        self.parent.title("PASTA Time Series Model")
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=1)
        self.parent.rowconfigure(2)
        self.parent.columnconfigure(0, weight=1)

        #
        # Observation Frame (Frame1)
        #

        Frame1 = ttk.Labelframe(self.parent, text='Observations')
        Frame1.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        Frame1.columnconfigure(1, weight=1)
        Frame1.rowconfigure(3, weight=2)

        #ibtn = tk.Button(Frame1, text="Read Series", command=self.read_series)
        #ibtn.grid(row=1, column=0, padx=5, pady=5)

        f = Figure(facecolor="white", figsize=(1, 1))
        self.ts_ax = f.add_subplot(111)
        #self.ts_ax.set_position([0.075, 0.1, 0.9, 0.85])  # This should not be hardcoded
        self.ts_canvas = FigureCanvasTkAgg(f, master=Frame1)
        self.ts_canvas.get_tk_widget().grid(row=1, column=1, columnspan=2,
                                            rowspan=4,
                                            sticky=tk.W + tk.E + tk.N + tk.S)
        # plot the observations
        self.obs_graph, = self.ts_ax.plot(self.ml.oseries.index,
                                             self.ml.oseries.values,
                                             color='black', marker='.',
                                             linestyle='')
        self.ts_ax.relim()
        self.ts_ax.autoscale_view()
        self.ts_canvas.show()
        sim=self.ml.simulate()
        self.sim_graph, = self.ts_ax.plot(sim.index, sim.values, color='blue')


        #
        # Stresses Frame (Frame2)
        #

        Frame2 = ttk.Labelframe(self.parent, text='Stresses')
        Frame2.grid(row=1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        Frame2.columnconfigure(1, weight=1)
        Frame2.columnconfigure(2, weight=1)
        Frame2.rowconfigure(4, weight=1)

        # buttons on right side
        ibtn = tk.Button(Frame2, text="Plot Stress", command=self.plot_ir_series)
        ibtn.grid(row=0, column=0, columnspan=1, padx=5, pady=1, sticky=tk.W)

        # Table with time series components
        self.tree = ttk.Treeview(Frame2)
        self.tree.grid(row=0, column=1, rowspan=5, sticky=tk.W + tk.E + tk.N + tk.S)

        self.tree["columns"] = (0, 1, 2, 3)
        self.tree.column(0, width=60)
        self.tree.column(1, width=30)
        self.tree.column(2, width=30)
        self.tree.column(3, width=30)
        self.tree.heading(0, text="Imp. Resp.")
        self.tree.heading(1, text="P1")
        self.tree.heading(2, text="P2")
        self.tree.heading(3, text="P3")

        self.tree.bind("<<TreeviewSelect>>", self.select_ir)

        self.generateTreeView()

        self.popup = tk.Menu(self.parent, tearoff=0)
        self.popup.add_command(label="Delete", command=self.delete_ir)
        self.tree.bind("<Button-3>", self.popUpMenu)

        f = Figure(facecolor="white", figsize=(2, 1))
        self.ir_ax = f.add_subplot(111)
        #self.ir_ax.set_position([0.15, 0.15, 0.80, 0.8])
        self.ir_canvas = FigureCanvasTkAgg(f, master=Frame2)
        self.ir_canvas.get_tk_widget().grid(row=0, column=2, rowspan=5,
                                            sticky=tk.W + tk.E + tk.N + tk.S)
        self.ir_graph = None

        #
        # Button Frame (Frame3)
        #

        Frame3 = tk.Frame(self.parent)
        Frame3.grid(row=3, column=0, rowspan=5, sticky=tk.W + tk.E + tk.N + tk.S)

        Frame3.columnconfigure(1, weight=1)
        Frame3.columnconfigure(3, weight=1)

        ibtn = tk.Button(Frame3, text="Optimize", command=self.optimize)
        ibtn.grid(row=1, column=0, padx=5, pady=1)
        ibtn = tk.Button(Frame3, text="Show results", command=self.show_results)
        ibtn.grid(row=2, column=0, padx=5, pady=1)

    def popUpMenu(self, event):
        if self.tree.selection():
            self.popup.post(event.x_root, event.y_root)

    def delete_ir(self):
        for name in self.tree.selection():
            self.ml.tseriesdict.pop(name)
        self.generateTreeView()

    def select_ir(self, event):
        if self.ir_graph != None:
            self.ir_ax.clear()
            self.ir_graph = None

        # draw graph of selection
        for name in self.tree.selection():
            ts = self.ml.tseriesdict[name]
            if hasattr(ts, 'rfunc'):
                ir = ts.rfunc
                p = ts.parameters['initial']
                s = ir.block(p)

                self.ir_graph, = self.ir_ax.plot(s)
                self.ir_ax.relim()
                self.ir_ax.autoscale_view()
                #self.ir_canvas.show()
            self.ir_canvas.show()

    def generateTreeView(self):
        for ch in self.tree.get_children():
            self.tree.detach(ch)
        for key in self.ml.tseriesdict:
            ts=self.ml.tseriesdict[key]
            values = ts.parameters['initial']
            valueStr = []
            for v in values:
                if v >= 1000:
                    valueStr.append('{:.0f}'.format(v))
                elif v >= 100:
                    valueStr.append('{:.1f}'.format(v))
                else:
                    valueStr.append('{:.2f}'.format(v))
            # valueStr = [Gamma.__name__] + valueStr
            if hasattr(ts, 'rfunc'):
                valueStr = [ts.rfunc.__class__.__name__] + valueStr
            else:
                valueStr = ['Constant'] + valueStr
            self.tree.insert('', 'end', text=ts.name, values=valueStr, iid=ts.name)

    def plot_ir_series(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        for name in self.tree.selection():
            ts=self.ml.tseriesdict[name]
            if not ts.stress.empty:
                ts.stress.plot(ax=ax)
        plt.show()

    def optimize(self):
        self.parent.config(cursor="wait")

        # solve
        self.ml.solve()

        # set optimal parameters as initial parameters
        for name in self.ml.tseriesdict:
            ts=self.ml.tseriesdict[name]
            ts.parameters['initial'] = self.ml.get_parameters(name)

        # change parameters in Treeview
        self.generateTreeView()

        # show graph
        sim = self.ml.simulate()
        if self.sim_graph != None:
            self.sim_graph.remove()
        self.sim_graph, = self.plot_simulation(sim,self.ts_ax)
        self.ts_ax.relim()
        self.ts_ax.autoscale_view()
        self.ts_canvas.show()

        self.parent.config(cursor="")

    def plot_simulation(self,sim,ax):
        sim_graph, = ax.plot(sim.index, sim.values, color='blue')
        return sim_graph

    def show_results(self):
        self.ml.plots.plot_results()

    def load_settings(self):
        if os.path.isfile(self.settingsFile):
            fileObject = open(self.settingsFile, 'r')
            self.settings = pickle.load(fileObject)
            fileObject.close()
        else:
            self.settings = {}

    def save_settings(self):
        fileObject = open(self.settingsFile, 'wb')
        pickle.dump(self.settings, fileObject)

    def not_yet_implemented(self):
        tkMessageBox.showerror('Not yet implemented', 'Working on it...')


    def center(self, root, w, h):
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        return "%dx%d+%d+%d" % (w, h, (sw - w) / 2, (sh - h) / 2)
