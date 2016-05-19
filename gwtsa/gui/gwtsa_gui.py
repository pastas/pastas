# -*- coding: utf-8 -*-
"""
Created on Wed May 18 20:47:20 2016

@author: ruben
"""

from Tkinter import Tk, W, N, E, S
from ttk import Button, Label,  Frame, Labelframe, Treeview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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
        
        ibtn = Button(Frame1, text="Read file", command=self.load_file)
        ibtn.grid(row=1, column=0, padx=5, pady=5)
        if True:
            f = Figure(facecolor="white",figsize=(2,1))
            self.ts_ax = f.add_subplot(111)
            self.ts_canvas = FigureCanvasTkAgg(f, master=Frame1)
            #self.ts_canvas.show()
            self.ts_canvas.get_tk_widget().grid(row=1, column=1, columnspan=2, rowspan=4,
                sticky = W+E+N+S)
        self.observation_graph=None
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
        
        ibtn = Button(Frame2, text="Read file", command=self.load_file)
        ibtn.grid(row=0, column=0, padx=5, pady=1)
        ibtn = Button(Frame2, text="Download", command=self.load_file)
        ibtn.grid(row=1, column=0, padx=5, pady=1)
        ibtn = Button(Frame2, text="Plot series", command=self.load_file)
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
        self.tree.bind("<Button-1>", self.select_ir)

        if True:
            # test-data
            self.tree.insert('', 'end', 'recharge', text='Recharge',
                        values=["Tseries2"])
            self.tree.insert('recharge', 'end', text='Precipitation',
                        values=['Gamma',"0.3","0.2","10"])
            self.tree.insert('recharge', 'end', text='Evaporation',
                        values=['Gamma',"0.3","0.2","5"])
                        
            self.tree.insert('', 'end', 'drainage_level', text='Drainage level',
                        values=['Constant',"1.02"])
            self.tree.insert('', 'end', 'well_1', text='Well 1',
                        values=['Hantush',"-","-","-"], tags = ('disabled',))
        
        if True:
            f = Figure(facecolor="white",figsize=(1,1))
            self.ir_ax = f.add_subplot(111)
            self.ir_canvas = FigureCanvasTkAgg(f, master=Frame2)
            #self.ir_canvas.show()
            self.ir_canvas.get_tk_widget().grid(row=0, column=2, rowspan=4,
                sticky = W+E+N+S)
        
        #%% Frame 3
        Frame3 = Frame(self.parent)
        Frame3.grid(row=3,column=0, sticky = W+E+N+S)
        
        Frame3.columnconfigure(1, weight=1)
        Frame3.columnconfigure(3, weight=1) 
        
        ibtn = Button(Frame3, text="Optimize", command=self.load_file)
        ibtn.grid(row=1, column=0, padx=5, pady=1)
        ibtn = Button(Frame3, text="Show results", command=self.load_file)
        ibtn.grid(row=2, column=0, padx=5, pady=1)
        
        ibtn = Button(Frame3, text="Simulate", command=self.load_file)
        ibtn.grid(row=1, column=2, padx=5, pady=1)        
        
        ibtn = Button(Frame3, text="Save to file", command=self.load_file)
        ibtn.grid(row=1, column=4, padx=5, pady=1)
        ibtn = Button(Frame3, text="Export", command=self.load_file)
        ibtn.grid(row=2, column=4, padx=5, pady=1)
    
    def select_ir(self,event):
        item = self.tree.identify('item',event.x,event.y)
        print("you clicked on", self.tree.item(item,"text"))
        values=self.tree.item(item,"values")        
        
        ir = getattr(gwtsa.rfunc, values[0])()
        #s=ir.block(values[1:ir.nparam])
        s=ir.block([1,2,3])
        self.ir_ax.plot(s)
        self.ir_ax.relim()            
        self.ir_ax.autoscale_view()
        self.ir_canvas.show()
    
    def load_file(self):
        if self.settings.has_key('observation_file'):
            dlg = tkFileDialog.Open(initialdir=self.settings['observation_file'])
        else:
            dlg = tkFileDialog.Open()
        fname = dlg.show()
        if fname != '':
            print(fname)
            self.settings['observation_file'] = fname
            self.save_settings()
            series = ImportSeries(fname,'dino')
            
            if self.observation_graph!=None:
                self.observation_graph.remove()
            #self.observation_graph=dino.stand.plot(ax=self.ts_ax)
            self.observation_graph,=self.ts_ax.plot(series.series.index,
                                                    series.series.values,
                                                    color='black',marker='.')
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



