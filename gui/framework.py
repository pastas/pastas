"""
This file contains the class to create the top Tkinter menu. Inspired by
Bashkar Chaudhary's book on Tkinter (2013) and adapted by R.A. Collenteur.
"""

from Tkinter import *


class Framework(object):
    def __init__(self, parent):
        self.parent = parent

    def create_menu(self, menulist):
        """

        Parameters
        ----------
        menulist: tuple
            tuple list containing the menu entries in the following format:
            ('Filemenu-label/icon/accelerator/command')

        Returns
        -------

        """
        self.icon = []

        self.menubar = Menu(self.parent, tearoff=0)

        for item in menulist:
            menu = Menu(self.menubar, tearoff=0)
            label, items = item.split('-')

            items = items.split(',')
            for menuitem in items:
                self.add_menuitem(menu, menuitem)

            self.menubar.add_cascade(label=label, menu=menu)
        self.parent.config(menu=self.menubar)

    def add_menuitem(self, menu, item):
        if item == 'Sep':
            menu.add_separator()
        else:
            label, icon, acc, cmd = item.split('/')  # Unpack item
            try:
                icon = PhotoImage(file="icons/16/%s.png" % icon)
            except:
                icon = ''
            self.icon.append(icon) # Icons need to stored in self to work..
            if cmd == '':
                cmd = 'self.donothing'

            menu.add_command(label=label, image=self.icon[-1], accelerator=acc,
                             compound=LEFT, command=eval(cmd))

    def donothing(self):
        filewin = Toplevel()
        button = Button(filewin, text="Do nothing button")
        button.pack()
