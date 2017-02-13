"""
This file contains the classes that can be used to import groundwater level
data from dinoloket.nl.

TODO: Get rid of filternummer en opmerking in self.series

"""

import numpy as np
import pandas as pd
from pastas.read.datamodel import DataModel


class dinodata(DataModel):
    def __init__(self, fname):
        """This method can be used to import files from Dinoloket that contain
         groundwater level measurements (https://www.dinoloket.nl/)

        Parameters
        ----------
        fname: str
            Filename and path to a Dino file.

        Returns
        -------
        DataModel: object
            returns a standard Pastas DataModel object.

        """

        DataModel.__init__(self)

        # Read the file
        dino = DinoGrondwaterstand(fname)

        self.data = dino.data
        self.x = dino.x
        self.y = dino.y
        self.latlon = self.rd2wgs(dino.x, dino.y)
        self.metadata = dino.meta[-1]


class DinoGrondwaterstand:
    def __init__(self, fname):
        with open(fname, 'r') as f:
            # lees de header
            line = f.readline()
            header = dict()
            while line not in ['\n', '', '\r\n']:
                propval = line.split(',')
                prop = propval[0]
                prop = prop.replace(':', '')
                prop = prop.strip()
                val = propval[1]
                if propval[2] != '':
                    val = val + ' ' + propval[2].replace(':', '') + ' ' + \
                          propval[3]
                header[prop] = val
                line = f.readline()

            # lees gat
            while (line == '\n') or (line == '\r\n'):
                line = f.readline()

            # lees referentieniveaus
            ref = dict()
            while line not in ['\n', '', '\r\n']:
                propval = line.split(',')
                prop = propval[0]
                prop = prop.replace(':', '')
                prop = prop.strip()
                if len(propval) > 1:
                    val = propval[1]
                    ref[prop] = val
                line = f.readline()

            # lees gat
            while (line == '\n') or (line == '\r\n'):
                line = f.readline()

            # lees meta-informatie
            metaList = list()
            line = line.strip()
            properties = line.split(',')
            line = f.readline()
            while line not in ['\n', '', '\r\n']:
                meta = dict()
                line = line.strip()
                values = line.split(',')
                for i in range(0, len(values)):
                    meta[properties[i]] = values[i]
                metaList.append(meta)
                line = f.readline()

            # lees gat
            while (line == '\n') or (line == '\r\n'):
                line = f.readline()

            line = line.strip()
            titel = line.split(',')
            while '' in titel:
                titel.remove('')

            # lees reeksen
            if line != '':
                # Validate if titles are valid names
                validator = np.lib._iotools.NameValidator()
                titel = validator(titel)
                dtype = [np.float64] * (len(titel))
                dtype[0] = "S11"
                dtype[1] = np.int
                dtype[2] = "S10"
                dtype[titel.index('Bijzonderheid')] = object
                dtype[titel.index('Opmerking')] = object
                dtype = list(zip(titel, dtype))

                usecols = range(0, len(titel))
                # # usecols.remove(2)
                measurements = pd.read_csv(f, header=None, names=titel,
                                           parse_dates=['Peildatum'],
                                           index_col='Peildatum',
                                           dayfirst=True,
                                           usecols=usecols)
                ts = measurements['Stand_cm_tov_NAP'] / 100
                #
                # measurements = np.genfromtxt(f, delimiter=',',
                #                              dtype=None,
                #                              usecols=range(0, len(titel)),
                #                              names=titel,
                #                              missing_values=[''])
                # values = measurements['Stand_cm_tov_NAP'] / float(100)
                # values[values == -0.01] = np.NAN

                # measurements = np.genfromtxt(fname, delimiter=',', dtype=dtype,
                #                              names=titel, usecols=usecols)
                # values = measurements['Stand_cm_tov_NAP'] / 100

                # %% zet de ingelezen data om in een reeks
                # if measurements['Stand_cm_tov_NAP'].dtype == np.dtype('bool'):
                #     # wanneer bleek dat het allemaal lege waarden waren
                #     if values.size == 1:
                #         values = np.NAN
                #     else:
                #         values[:] = np.NAN

                # if measurements['Peildatum'].size == 1:
                #     datum = pd.to_datetime(
                #         measurements['Peildatum'].item(), dayfirst=True)
                #     ts = pd.Series([values], [datum])
                # else:
                #     datum = pd.to_datetime(measurements['Peildatum'])
                #     ts = pd.Series(values, datum)
            else:
                measurements = None
                ts = pd.Series()

            # %% kies welke invoer opgeslagen wordt
            self.meta = metaList
            if self.meta:
                self.locatie = self.meta[-1]['Locatie']
                self.filternummer = int(float(self.meta[-1]['Filternummer']))
                self.x = float(self.meta[-1]['X-coordinaat'])
                self.y = float(self.meta[-1]['Y-coordinaat'])
                meetpunt = self.meta[-1]['Meetpunt (cm t.o.v. NAP)']
                if meetpunt == '':
                    self.meetpunt = np.nan
                else:
                    self.meetpunt = float(meetpunt) / 100
                maaiveld = self.meta[-1]['Maaiveld (cm t.o.v. NAP)']
                if maaiveld == '':
                    self.maaiveld = np.nan
                else:
                    self.maaiveld = float(maaiveld) / 100
                bovenkant_filter = self.meta[-1][
                    'Bovenkant filter (cm t.o.v. NAP)']
                if bovenkant_filter == '':
                    self.bovenkant_filter = np.nan
                else:
                    self.bovenkant_filter = float(bovenkant_filter) / 100
                self.onderkant_filter = self.meta[-1][
                    'Onderkant filter (cm t.o.v. NAP)']
                if self.onderkant_filter == '':
                    self.onderkant_filter = np.nan
                else:
                    self.onderkant_filter = float(self.onderkant_filter) / 100
            else:
                # de metadata is leeg
                self.locatie = ''
                self.filternummer = np.nan
                self.x = np.nan
                self.y = np.nan
                self.meetpunt = np.nan
                self.maaiveld = np.nan
                self.bovenkant_filter = np.nan
                self.onderkant_filter = np.nan
            self.data = measurements
            self.stand = ts

        f.close()
