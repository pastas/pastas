# Import the necessary modules required
import sys

from PySide.QtGui import QApplication, QMainWindow, QIcon, QStatusBar, QLabel, \
    QMessageBox, QAction


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # Set main window properties
        self.setWindowTitle("Pastas Time Series Analysis")
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowIcon(QIcon("/images/"))

        # Create main window features
        self.create_statusbar()
        self.create_menu()

    def create_statusbar(self):
        self.statusbar = QStatusBar()
        self.statusbar.showMessage("This is the status of pasta", 0)
        self.statuslabel = QLabel("program status:")
        self.statusbar.addWidget(self.statuslabel)

        self.setStatusBar(self.statusbar)

    def create_menu(self):
        # Create menu actions
        self.create_menu_actions()

        # Create file menu
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.closeAction)
        self.fileMenu.addSeparator()

        # Create edit menu
        self.editMenu = self.menuBar().addMenu("&Edit")

        # Create help menu
        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.aboutAction)
        self.helpMenu.addAction(self.helpAction)


    def create_menu_actions(self):
        self.closeAction = QAction('C&lose', self, triggered=self.exit_program)
        self.aboutAction = QAction('A&bout', self, triggered=self.about_program)

        self.helpAction = QAction('H&elp', self, triggered=self.help_program)

    def exit_program(self):
        msg = "Are you sure you want to close this window?"
        #answer = QMessageBox.
        self.close()

    def about_program(self):
        QMessageBox.about(self, "About Simple Text Editor",
                          "This example demonstrates the use "\
                          "of Menu Bar")

    def help_program(self):
        QMessageBox.about(self, "help", "Help not supported yet.")












if __name__ == '__main__':
    # Exception Handling
    try:
        myApp = QApplication(sys.argv)
        mainWindow = MainWindow()
        mainWindow.show()
        myApp.exec_()
        sys.exit(0)
    except NameError:
        print("Name Error:", sys.exc_info()[1])
    except SystemExit:
        print("Closing Window...")
    except Exception:
        print(sys.exc_info()[1])
