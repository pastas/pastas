# -*- coding: utf-8 -*-
"""
Created on Wed May 18 20:47:20 2016

@author: ruben
"""

from Tkinter import Tk, W, N, E, S
from ttk import Button, Label,  Frame, Labelframe, Treeview
import tkMessageBox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkFileDialog
from gwtsa.imports.import_series import ImportSeries
import gwtsa
import pickle
import os.path

class GwtsaGui(Frame):
    def __init__(self,parent):
        Frame.__init__(self, parent)
        
        self.settingsFile = 'settings'
        self.load_settings()
        
        self.obs = None
        self.irs = [];
        
        self.parent = parent
        self.initUI()
        
    def initUI(self):        
        self.parent.title("GWTSA")
        #self.pack(fill=BOTH, expand=True)
        
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=1)
        self.parent.rowconfigure(2)
        self.parent.columnconfigure(0, weight=1)
        
        #%% Frame 1
        Frame1 = Labelframe(self.parent, text='Observations')
        Frame1.grid(row=0,column=0,sticky = W+E+N+S)
        #container.pack(fill=BOTH, expand=True)
        
        Frame1.columnconfigure(1, weight=1)
        Frame1.rowconfigure(3, weight=2)
        
        ibtn = Button(Frame1, text="Read obs. file", command=self.load_file)
        ibtn.grid(row=1, column=0, padx=5, pady=5)
        if True:
            f = Figure(facecolor="white",figsize=(2,1))
            self.ts_ax = f.add_subplot(111)
            self.ts_ax.set_position([0.075,0.1,0.9,0.85])
            self.ts_canvas = FigureCanvasTkAgg(f, master=Frame1)
            #self.ts_canvas.show()
            self.ts_canvas.get_tk_widget().grid(row=1, column=1, columnspan=2, rowspan=4,
                sticky = W+E+N+S)
        self.obs_graph=None
        self.sim_graph=None
        #area = Text(container)
        #area.grid(row=1, column=1, columnspan=2, rowspan=4, 
        #    padx=5, pady=5, sticky=E+W+S+N)
        
        #%% Frame 2
        Frame2 = Labelframe(self.parent, text='Stresses')
        Frame2.grid(row=1,column=0, sticky = W+E+N+S)
        lbl = Label(Frame2, text="Stresses")
        lbl.grid(sticky=W, pady=4, padx=5)
        
        Frame2.columnconfigure(1, weight=1)
        Frame2.columnconfigure(2, weight=1)
        Frame2.rowconfigure(3, weight=1)
        
        ibtn = Button(Frame2, text="Read str. file", command=self.load_ir_file)
        ibtn.grid(row=0, column=0, padx=5, pady=1)
        ibtn = Button(Frame2, text="Download", command=self.not_yet_implemented)
        ibtn.grid(row=1, column=0, padx=5, pady=1)
        ibtn = Button(Frame2, text="Plot series", command=self.plot_ir_series)
        ibtn.grid(row=2, column=0, padx=5, pady=1)
        
        self.tree = Treeview(Frame2, height=4)
        self.tree.grid(row=0, column=1, rowspan=4, sticky = W+E+N+S)        
        
        self.tree["columns"]=("three","four","five","six")
        self.tree.column("#0",width=100)
        #tree.column("two", width=30)
        self.tree.column("three", width=45)
        self.tree.column("four", width=15)
        self.tree.column("five", width=15)
        self.tree.column("six", width=15)
        #tree.heading("two", text="Active")
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
                        values=['Gamma',"500","1","100"])
            self.tree.insert('recharge', 'end', text='Evaporation',
                        values=['Factor',"0.79"])
                        
            self.tree.insert('', 'end', text='Drainage level',
                        values=['Constant',"1.02"])
            self.tree.insert('', 'end', text='Well 1',
                        values=['Hantush',"1","2","3"], tags = ('disabled',))
        else:
            self.tree.insert('', 'end', 'drainage_level', text='Drainage level',
            values=['Constant',"1.02"])
            self.irs.append(None)
        
        if True:
            f = Figure(facecolor="white",figsize=(1,1))
            self.ir_ax = f.add_subplot(111)
            self.ir_ax.set_position([0.25,0.1,0.7,0.85])
            self.ir_canvas = FigureCanvasTkAgg(f, master=Frame2)
            #self.ir_canvas.show()
            self.ir_canvas.get_tk_widget().grid(row=0, column=2, rowspan=4,
                sticky = W+E+N+S)
        self.ir_graph=None
        
        #%% Frame 3
        Frame3 = Frame(self.parent)
        Frame3.grid(row=3,column=0, sticky = W+E+N+S)
        
        Frame3.columnconfigure(1, weight=1)
        Frame3.columnconfigure(3, weight=1) 
        
        ibtn = Button(Frame3, text="Optimize", command=self.optimize)
        ibtn.grid(row=1, column=0, padx=5, pady=1)
        ibtn = Button(Frame3, text="Show results", command=self.not_yet_implemented)
        ibtn.grid(row=2, column=0, padx=5, pady=1)
        
        ibtn = Button(Frame3, text="Simulate", command=self.not_yet_implemented)
        ibtn.grid(row=1, column=2, padx=5, pady=1)        
        
        ibtn = Button(Frame3, text="Save to file", command=self.not_yet_implemented)
        ibtn.grid(row=1, column=4, padx=5, pady=1)
        ibtn = Button(Frame3, text="Export", command=self.not_yet_implemented)
        ibtn.grid(row=2, column=4, padx=5, pady=1)
    
    def select_ir(self,event):
        if self.ir_graph!=None:
            self.ir_graph.remove()
            self.ir_graph=None
        
        print(self.tree.selection())
        values=self.tree.item(self.tree.selection(),"values")
        #tag=self.tree.item(self.tree.selection(),"tag")
        
        if values[0]=='Constant':
            pass
        elif values[0]=='Tseries2':
            pass
        elif values[0]=='Factor':
            # get the brother from the parent
            pass
        else:
            ir = getattr(gwtsa.rfunc, values[0])()
            #s=ir.block([1,2,3])
            p=list(values[1:ir.nparam+1])
            p=[float(x) for x in p]
            s=ir.block(p)
            
            self.ir_graph,=self.ir_ax.plot(s)
            self.ir_ax.relim()            
            self.ir_ax.autoscale_view()
            self.ir_canvas.show()
    
    def load_file(self):
        if self.settings.has_key('observation_file'):
            dlg = tkFileDialog.Open(initialfile=self.settings['observation_file'])
        else:
            dlg = tkFileDialog.Open()
        fname = dlg.show()
        if fname != '':
            print(fname)
            self.settings['observation_file'] = fname
            self.save_settings()
            self.obs = ImportSeries(fname,'dino')
            
            if self.obs_graph!=None:
                self.obs_graph.remove()
            self.obs_graph,=self.ts_ax.plot(self.obs.series.index,
                                                    self.obs.series.values,
                                                    color='black',marker='.')
            self.ts_ax.relim()            
            self.ts_ax.autoscale_view()
            self.ts_canvas.show()
            
    def load_ir_file(self):
        if self.settings.has_key('ir_file'):
            dlg = tkFileDialog.Open(initialfile=self.settings['ir_file'])
        else:
            dlg = tkFileDialog.Open()
        irfname = dlg.show()
        if irfname != '':
            print(irfname)
            self.settings['ir_file'] = irfname
            self.save_settings()
            series = ImportSeries(irfname,'knmi',variable='RH')
            self.irs.append(series)
            self.tree.insert('', 'end', text='Precipitation',
                        values=['Gamma',"500","1","100"])
            #self.tree.insert('recharge', 'end', text='Evaporation',
            #            values=['Factor',"0.79"])
            if False:
                series = ImportSeries(irfname,'knmi',variable='EV24')
                self.irs.append(series)
                self.tree.insert('', 'end', text='Evaporation',
                        values=['Gamma',"500","1","100"])
                        
    def plot_ir_series(self):
        f=plt.figure()
        ax = f.add_subplot(111)
        for ir in self.irs:
            if ir != None:
                ir.series.plot(ax = ax)
        plt.show()
    
    def optimize(self):
        if self.obs == None:
            tkMessageBox.showerror('No observation series', 'First read observations')
        else:
            self.ml = gwtsa.Model(self.obs.series)
            chs=self.tree.get_children()
            for ch in chs:
                values=self.tree.item(ch,"values")
                print(values)
                
                if values[0]=='Constant':
                    ts=gwtsa.Constant(value=float(values[1]))
                    self.ml.addtseries(ts)
                elif values[0]=='Tseries2':
                    # not implemented yet
                    pass
                else:
                    stress=self.irs[self.tree.index(ch)].series
                    rfunc = getattr(gwtsa.rfunc, values[0])()
                    ts = gwtsa.Tseries(stress, rfunc, name=values[0])
                    self.ml.addtseries(ts)
                    
            # solve
            self.ml.solve()
            
            # change parameters in Treeview
            for i in range(len(chs)):
                ch=chs[i]
                ts=self.ml.tserieslist[i]
                value=ts.parameters['value']
                # convert to strings
                value=['{:.2f}'.format(x) for x in value]
                #value=['{:.3g}'.format(x) for x in value]
                
                # add to existing name
                values=self.tree.item(ch,"values")
                values=[values[0]]+value
                # set values
                self.tree.item(ch,values=values)
                
            
            # show graph
            h = self.ml.simulate()
            if self.sim_graph!=None:
                self.sim_graph.remove()
            self.sim_graph,=self.ts_ax.plot(h.index,h.values,color='red')
            self.ts_ax.relim()            
            self.ts_ax.autoscale_view()
            self.ts_canvas.show()
    
    def load_settings(self):
        if os.path.isfile(self.settingsFile):
            fileObject = open(self.settingsFile,'r')
            self.settings = pickle.load(fileObject)
            fileObject.close()
        else:
            self.settings={}
            
    def save_settings(self):
        fileObject = open(self.settingsFile,'wb')
        pickle.dump(self.settings,fileObject)
    
    def not_yet_implemented(self):
        tkMessageBox.showerror('Not yet implemented', 'Working on it...')

def center(toplevel):
    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

def main():
  
    root = Tk()
    root.geometry("600x500+300+300")
    GwtsaGui(root)
    root.mainloop()  

if __name__ == '__main__':
    main()  



