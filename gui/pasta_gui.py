"""
This file contains the GUI made for pasta.

"""

from Tkinter import Tk, W, N, E, S, Toplevel, StringVar, Menu, BooleanVar
from ttk import Button, Label, Frame, Labelframe, Treeview, OptionMenu, Entry
import tkMessageBox
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkFileDialog
from pastas import *
import pickle
import os.path
from framework import Framework


class PastaGui(Frame, Framework):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        Framework.__init__(self, parent)

        self.settingsFile = 'settings'
        self.load_settings()

        self.obs = None
        self.irs = []
        self.tserieslist = []  # Empty list to store all tseries objects
        self.modellist = []  # Empty list to store all the optimized models

        self.parent = parent
        self.build_menu()
        self.initUI()

    def initUI(self):
        # Initialize the GUI Application and create all the frames

        self.parent.title("PASTA Time Series Model")
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=1)
        self.parent.rowconfigure(2)
        self.parent.columnconfigure(0, weight=1)

        #
        # %% Observation Frame (Frame1)
        #

        Frame1 = Labelframe(self.parent, text='Observations')
        Frame1.grid(row=0, column=0, sticky=W + E + N + S)
        Frame1.columnconfigure(1, weight=1)
        Frame1.rowconfigure(3, weight=2)

        ibtn = Button(Frame1, text="Read Series", command=self.read_series)
        ibtn.grid(row=1, column=0, padx=5, pady=5)

        f = Figure(facecolor="white", figsize=(1, 1))
        self.ts_ax = f.add_subplot(111)
        self.ts_ax.set_position([0.075, 0.1, 0.9, 0.85])  # This should not be hardcoded
        self.ts_canvas = FigureCanvasTkAgg(f, master=Frame1)
        self.ts_canvas.get_tk_widget().grid(row=1, column=1, columnspan=2,
                                            rowspan=4,
                                            sticky=W + E + N + S)

        self.series_graph = None
        self.sim_graph = None

        #
        # %% Stresses Frame (Frame2)
        #

        Frame2 = Labelframe(self.parent, text='Stresses')
        Frame2.grid(row=1, column=0, sticky=W + E + N + S)
        Label(Frame2, text="Stresses").grid(sticky=W, pady=4, padx=5)

        Frame2.columnconfigure(1, weight=1)
        Frame2.columnconfigure(2, weight=1)
        Frame2.rowconfigure(4, weight=1)

        ibtn = Button(Frame2, text="Read Stress", command=self.load_ir_file)
        ibtn.grid(row=0, column=0, columnspan=1, padx=5, pady=1, sticky=W)
        ibtn = Button(Frame2, text="Download", command=self.not_yet_implemented)
        ibtn.grid(row=1, column=0, columnspan=1, padx=5, pady=1, sticky=W)
        ibtn = Button(Frame2, text="Plot Stress", command=self.plot_ir_series)
        ibtn.grid(row=2, column=0, columnspan=1, padx=5, pady=1, sticky=W)
        ibtn = Button(Frame2, text='Add Stress', command=self.add_stress)
        ibtn.grid(row=3, column=0, columnspan=1, padx=5, pady=1, sticky=W)

        # Table with time series components

        self.tree = Treeview(Frame2, height=4)
        self.tree.grid(row=0, column=1, rowspan=5, sticky=W + E + N + S)

        self.tree["columns"] = ('three', 'four', 'five', 'six')
        self.tree.column("#0", width=100)
        # self.tree.column("two", width=30)
        self.tree.column("three", width=45)
        self.tree.column("four", width=15)
        self.tree.column("five", width=15)
        self.tree.column("six", width=15)
        # tree.heading("two", text="Active")
        self.tree.heading("three", text="Imp. Resp.")
        self.tree.heading("four", text="Par 1")
        self.tree.heading("five", text="Par 2")
        self.tree.heading("six", text="Par 3")

        self.tree.tag_configure('disabled', foreground='grey')
        self.tree.bind("<<TreeviewSelect>>", self.select_ir)

        if False:
            # test-data
            self.tree.insert('', 'end', 'recharge', text='Recharge',
                             values=["Tseries2"])
            self.tree.insert('recharge', 'end', text='Precipitation',
                             values=['Gamma', "500", "1", "100"])
            self.tree.insert('recharge', 'end', text='Evaporation',
                             values=['Factor', "0.79"])

            self.tree.insert('', 'end', text='Drainage level',
                             values=['Constant', "1.02"])
            self.tree.insert('', 'end', text='Well 1',
                             values=['Hantush', "1", "2", "3"], tags=('disabled',))
        else:
            # self.tree.insert('', 'end', 'drainage_level', text='Drainage level',
            #                 values=['Constant', "1.02"])
            # self.irs.append(None)
            # ts = Constant(value=1.02,name='Drainage level')
            # self.tserieslist.append(ts)
            pass

        self.generateTreeView()

        self.popup = Menu(self.parent, tearoff=0)
        # self.popup.add_command(label="Disable", command=self.disable_ir)
        self.ir_is_disabled = BooleanVar()

        self.popup.add_checkbutton(label="Disable", onvalue=True, offvalue=False,
                                   variable=self.ir_is_disabled, command=self.disable_ir)
        self.popup.add_command(label="Delete", command=self.delete_ir)
        self.tree.bind("<Button-3>", self.popUpMenu)

        f = Figure(facecolor="white", figsize=(2, 1))
        self.ir_ax = f.add_subplot(111)
        self.ir_ax.set_position([0.15, 0.15, 0.80, 0.8])
        self.ir_canvas = FigureCanvasTkAgg(f, master=Frame2)
        self.ir_canvas.get_tk_widget().grid(row=0, column=2, rowspan=4,
                                            sticky=W + E + N + S)
        self.ir_graph = None

        # %% Frame 3
        Frame3 = Frame(self.parent)
        Frame3.grid(row=3, column=0, sticky=W + E + N + S)

        Frame3.columnconfigure(1, weight=1)
        Frame3.columnconfigure(3, weight=1)

        ibtn = Button(Frame3, text="Optimize", command=self.optimize)
        ibtn.grid(row=1, column=0, padx=5, pady=1)
        ibtn = Button(Frame3, text="Show results", command=self.show_results)
        ibtn.grid(row=2, column=0, padx=5, pady=1)

        ibtn = Button(Frame3, text="Simulate", command=self.not_yet_implemented)
        ibtn.grid(row=1, column=2, padx=5, pady=1)

        ibtn = Button(Frame3, text="Save to file", command=self.not_yet_implemented)
        ibtn.grid(row=1, column=4, padx=5, pady=1)
        ibtn = Button(Frame3, text="Export", command=self.not_yet_implemented)
        ibtn.grid(row=2, column=4, padx=5, pady=1)

    def popUpMenu(self, event):
        if self.tree.selection():
            self.popup.post(event.x_root, event.y_root)

    def disable_ir(self):
        ts = self.tserieslist[self.tree.index(self.tree.selection())]
        if ts.disabled:
            ts.disabled = False
        else:
            ts.disabled = True
        self.generateTreeView()
        # if ts.disabled:
        #     self.ir_is_disabled = True
        # tags=self.tree.item(self.tree.selection(),"tags")
        # if 'disabled' in tags:
        #     self.tree.item(self.tree.selection()[0],tags = ())
        # else:
        #     self.tree.item(self.tree.selection()[0],tags = ('disabled',))

    def delete_ir(self):
        index = self.tree.index(self.tree.selection()[0])
        self.tserieslist.pop(index)
        self.generateTreeView()

    def select_ir(self, event):
        if self.ir_graph != None:
            self.ir_ax.clear()
            self.ir_graph = None

        # draw graph of selection
        ts = self.tserieslist[self.tree.index(self.tree.selection())]
        self.ir_is_disabled.set(ts.disabled)
        if hasattr(ts, 'rfunc'):
            ir = ts.rfunc
            p = ts.parameters['initial']
            s = ir.block(p)

            self.ir_graph, = self.ir_ax.plot(s)
            self.ir_ax.relim()
            self.ir_ax.autoscale_view()
            self.ir_canvas.show()
        self.ir_canvas.show()

    # Read the observed time series
    def read_series(self):
        if self.settings.has_key('observation_file'):
            dlg = tkFileDialog.Open(initialfile=self.settings['observation_file'])
        else:
            dlg = tkFileDialog.Open()
        fname = dlg.show()
        if fname != '':
            print(fname)
            self.settings['observation_file'] = fname
            self.save_settings()

            # TODO Make read_series choose the filetype e.g type='dino'
            self.oseries = ReadSeries(fname, 'dino')

            self.generateTreeView()

            # Automatically plot the oseries
            if self.series_graph != None:
                self.series_graph.remove()
            self.series_graph, = self.ts_ax.plot(self.oseries.series.index,
                                                 self.oseries.series.values,
                                                 color='black', marker='.',
                                                 linestyle='')
            self.ts_ax.relim()
            self.ts_ax.autoscale_view()
            self.ts_canvas.show()

    def build_menu(self):
        # Create the menu according to the specifications in the menulist.
        menulist = ('Bestand -Nieuw Project//Crtl+N/self.donothing,'
                    'Open/folder/Ctrl+O/self.donothing,'
                    'Opslaan.../save_as/Ctrl+S/self.donothing,'
                    'Opslaan als.../save_as/Ctrl+S/self.donothing,'
                    'Sep,'
                    'Sluit//Ctrl+Q/self.parent.quit',
                    'Edit -Maak ongedaan/undo/Ctrl+Z/self.donothing,'
                    'Sep',
                    'Project -Project Informatie/setting/Ctrl+I/self.donothing',
                    'Help -Help/help//self.donothing,'
                    'Over.../information//self.donothing')

        self.create_menu(menulist)

    def load_ir_file(self):
        if self.settings.has_key('ir_file'):
            dlg = tkFileDialog.Open(initialfile=self.settings['ir_file'])
        else:
            dlg = tkFileDialog.Open()
        irfname = dlg.show()
        if irfname != '':
            self.settings['ir_file'] = irfname
            self.save_settings()
            if True:
                series = ReadSeries(irfname, 'knmi', variable='RH')
                ts = Tseries(series.series, Gamma, name='Precipitation')
                ts.disabled = False
                self.tserieslist.append(ts)
            else:
                prec = ReadSeries(irfname, 'knmi', variable='RH')
                evap = ReadSeries(irfname, 'knmi', variable='EV24')
                ts = Recharge(prec.series, evap.series, Gamma, Linear, name='recharge')
                ts.disabled = False
                self.tserieslist.append(ts)

            self.generateTreeView()

    def generateTreeView(self):
        for ch in self.tree.get_children():
            self.tree.detach(ch)
        for ts in self.tserieslist:
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
            if ts.disabled:
                tags = ('disabled',)
            else:
                tags = ()
            self.tree.insert('', 'end', text=ts.name, values=valueStr, tags=tags)

    #
    # Add a stress through a seperate window
    #

    def add_stress(self):
        # Create a new pop-up window
        window = Toplevel(self.parent)
        # window.geometry('+300+300')
        top = Frame(window)
        top.grid(column=1, row=1, padx=10, pady=10)

        # Choose the stress type
        tseries_list = ['Tseries', 'Recharge']
        Label(top, text='Choose stress type:').grid(row=0, column=0,
                                                    sticky=W)
        tseries = StringVar(top)
        tseries.set(tseries_list[0])  # initial value
        # tseries.trace('w', self.update_stress_options())
        opt1 = OptionMenu(top, tseries, None, *tseries_list)
        opt1.grid(row=0, column=1, sticky=E)

        # Choose recharge model
        recharge_list = ['Linear', 'Preferential', 'Percolation',
                         'Combination']
        Label(top, text='Choose recharge model:').grid(row=1, column=0,
                                                       sticky=W)
        recharge = StringVar(top)
        recharge.set(recharge_list[0])
        opt2 = OptionMenu(top, recharge, recharge_list[0], *recharge_list)
        opt2.grid(row=1, column=1, sticky=E)

        # Choose Reponse function
        response_list = ['Gamma', 'Exponential']
        Label(top, text='Response function:').grid(row=2, column=0,
                                                   sticky=W)
        rfunc = StringVar(top)
        rfunc.set(response_list[0])
        opt2 = OptionMenu(top, rfunc, None, *response_list)
        opt2.grid(row=2, column=1, sticky=E)

        # Select Rain data
        precip = StringVar(top)
        r = Entry(top, textvariable=precip)
        Label(top, text='Select precipitation data:').grid(row=3, column=0,
                                                           sticky=W)
        Button(top, text='Browse', command=lambda: precip.set(
            tkFileDialog.askopenfilename())).grid(row=3, column=2, sticky=E)
        r.grid(row=3, column=1, sticky=W)

        # Select Evap data
        evap = StringVar(top)
        e = Entry(top, textvariable=evap)
        Label(top, text='Select evaporation data:').grid(row=4, column=0, sticky=W)
        Button(top, text='Browse', command=lambda: evap.set(
            tkFileDialog.askopenfilename())).grid(row=4, column=2, sticky=E)
        e.grid(row=4, column=1, sticky=W)

        # Lower button row
        cancel = Button(top, text='Cancel', command=window.destroy)  # Needs Fix,
        # windows.quit closes all windows now
        cancel.grid(row=5, column=1, sticky=S + W)
        save = Button(top, text='Add Stress', command=lambda: self.make_stress(
            tseries, precip, evap, rfunc, recharge))
        save.grid(row=5, column=2, sticky=S + E)

    def make_stress(self, tseries, precip, evap, rfunc, recharge):
        precip = ReadSeries(precip.get(), 'knmi', variable='RH')
        evap = ReadSeries(evap.get(), 'knmi', variable='EV24')
        rfunc = eval(rfunc.get())()
        recharge = eval(recharge.get())()
        ts = eval(tseries.get())(precip.series, evap.series, rfunc, recharge)
        self.tserieslist.append(ts)
        self.generateTreeView()

        # self.tree.insert()

    def plot_ir_series(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        for ts in self.tserieslist:
            if not ts.stress.empty:
                ts.stress.plot(ax=ax)
        plt.show()

    def optimize(self):
        if self.oseries == None:
            tkMessageBox.showerror('No observation series',
                                   'First read observations')
        else:
            self.parent.config(cursor="wait")

            # initialise the model
            self.ml = Model(self.oseries.series)

            # add tseries
            for ts in self.tserieslist:
                if not ts.disabled:
                    # TODO Make a deep copy
                    self.ml.add_tseries(ts)

            # solve
            self.ml.solve()

            # set optimal parameters as initial parameters
            for ts in self.tserieslist:
                if not ts.disabled:
                    ts.parameters['initial'] = self.ml.parameters['optimal'].loc[self.ml.parameters.name == ts.name]

            # change parameters in Treeview
            self.generateTreeView()

            # show graph
            h = self.ml.simulate()
            if self.sim_graph != None:
                self.sim_graph.remove()
            self.sim_graph, = self.ts_ax.plot(h.index, h.values, color='red')
            self.ts_ax.relim()
            self.ts_ax.autoscale_view()
            self.ts_canvas.show()

            self.parent.config(cursor="")

    def show_results(self):
        self.ml.plot_results()

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


def main():
    root = Tk()
    # w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    # root.geometry("%dx%d+0+0" % (w, h))
    # root.geometry("600x500+300+300")
    # sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    # w=600
    # h=500
    # root.geometry("%dx%d+%d+%d" % (w, h, (sw-w)/2, (sh-h)/2))
    root.geometry(center(root, 600, 500))
    PastaGui(root)
    root.mainloop()


def center(root, w, h):
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    return "%dx%d+%d+%d" % (w, h, (sw - w) / 2, (sh - h) / 2)


if __name__ == '__main__':
    main()
