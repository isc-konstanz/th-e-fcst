# -*- coding: utf-8 -*-
"""
    theforecast.database
    ~~~~~
    
    
"""
from abc import ABC, abstractmethod
from configparser import ConfigParser
import logging
import os
import datetime as dt
import pandas as pd
import pytz as tz

logger = logging.getLogger(__name__)


class Database(ABC):

    def __init__(self, timezone='UTC'):
        self.timezone = tz.timezone(timezone)

    @abstractmethod
    def get(self, keys, start, end, interval):
        """ 
        Retrieve data for a specified time interval of a set of data feeds
        
        :param keys: 
            the keys for which the values will be looked up for.
        :type keys: 
            list of str
        
        :param start: 
            the start time for which the values will be looked up for.
            For many applications, passing datetime.datetime.now() will suffice.
        :type start: 
            :class:`pandas.tslib.Timestamp` or datetime
        
        :param end: 
            the end time for which the data will be looked up for.
        :type end: 
            :class:`pandas.tslib.Timestamp` or datetime
        
        :param interval: 
            the time interval in seconds, retrieved values shout be returned in.
        :type interval: 
            int
        
        
        :returns: 
            the retrieved values, indexed in a specific time interval.
        :rtype: 
            :class:`pandas.DataFrame`
        """
        pass

    @abstractmethod
    def last(self, keys, interval):
        """
        Retrieve the last recorded values for a specified set of data feeds

        :param keys: 
            the keys for which the values will be looked up for.
        :type keys: 
            list of str
        
        :param interval: 
            the time interval in seconds, retrieved values shout be returned in.
        :type interval: 
            int
        """
        pass

    @abstractmethod
    def persist(self, data):
        """ 
        Store a set of data values, to persistently store them on the server
        
        :param data: 
            the data set to be persistently stored
        :type data: 
            :class:`pandas.DataFrame`
        """
        pass


class CsvDatabase(Database):

    def __init__(self, configs, timezone='UTC'):
        super().__init__(timezone=timezone)
        
        settingsfile = os.path.join(configs, 'database.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        self.output = settings.get('CSV', 'output')
        self.decimal = settings.get('CSV', 'decimal')
        self.separator = settings.get('CSV', 'separator')
        self.summarize = settings.getboolean('CSV', 'summarize')
        self.file = settings.get('CSV', 'file')
        
        self.datafile = os.path.join(settings.get('CSV', 'input'), self.file)
        if os.path.isfile(self.datafile):
            self.read_file(self.datafile, k=0, pred_start=100 * 1440)
            
    def get(self, keys, start, end, interval):
        if interval > 900:
            offset = (start - start.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() % interval
            data = self.data[keys].resample(str(int(interval)) + 's', base=offset).sum()
        else:
            data = self.data[keys]
        
        # TODO: Throw meaningful exception if requested time is not in available interval
        if start < end:
            end = end + dt.timedelta(seconds=interval)
            if end > data.index[-1]:
                end = data.index[-1]
            
            if start > data.index[0]:
                return data.loc[start:end]
            else:
                return data.loc[:end]
        else:
            return data.truncate(before=start).head(1)
        
    def last(self, keys, interval):
        date = dt.datetime.now(tz.utc).replace(second=0, microsecond=0)
        if date.minute % (interval / 60) != 0:
            date = date - dt.timedelta(minutes=date.minute % (interval / 60))
        
        return self.get(keys, date, date, interval)
    
    def persist(self, data):
        if data is not None:
            path = self.output
            self.concat_file(path, data)

    def read_file(self, path, k, pred_start, index_column='unixtimestamp', unix=True):
        """
        Reads the content of a specified CSV file.
        
        :param path: 
            the full path to the CSV file.
        :type path:
            str or unicode
        
        :param index_column: 
            the name of the column, that will be used as index. The index will be assumed 
            to be a time format, that will be parsed and localized.
        :type index_column:
            str or unicode
        
        :param unix: 
            the flag, if the index column contains UNIX timestamps that need to be parsed accordingly.
        :type unix:
            boolean
        
        :returns: 
            the retrieved columns, indexed by their date
        :rtype: 
            :class:`pandas.DataFrame`
        """
        
        csv = pd.read_csv(path, sep=self.separator, decimal=self.decimal, parse_dates=[0])
        
        if not csv.empty:  
            csv = csv.set_index('time')
            csv = csv.iloc[:pred_start + k]         
            self.data = csv
        
#         csv = pd.read_csv(path, sep=self.separator, decimal=self.decimal,
#                           index_col=index_column, parse_dates=[index_column])
# 
#         if not csv.empty:
#             if unix:
#                 csv.index = pd.to_datetime(csv.index, unit='ms')
#                 
#             csv.index = csv.index.tz_localize(tz.timezone('UTC')).tz_convert(self.timezone)
#         
#         csv.index.name = 'time'

    def read_nearest_file(self, date, path, index_column='time'):
        """
        Reads the daily content from a CSV file, closest to the passed date, 
        following the file naming scheme "YYYYMMDD*.filename"
        
        
        :param date: 
            the date for which a filename file will be looked up for.
        :type date: 
            :class:`pandas.tslib.Timestamp` or datetime
        
        :param path: 
            the directory, containing the filename files that should be looked for.
        :type path:
            str or unicode
        
        :param index_column: 
            the name of the column, that will be used as index. The index will be assumed 
            to be a time format, that will be parsed and localized.
        :type index_column:
            str or unicode
        
        :param timezone: 
            the timezone, in which the data is logged and available in the filename file.
            See http://en.wikipedia.org/wiki/List_of_tz_database_time_zones for a list of 
            valid time zones.
        :type timezone:
            str or unicode
        
        
        :returns: 
            the retrieved columns, indexed by their date
        :rtype: 
            :class:`pandas.DataFrame`
        """
        csv = None
        
        ref = int(date.strftime('%Y%m%d'))
        diff = 1970010100
        filename = None
        try:
            for f in os.listdir(path):
                if f.endswith('.csv'):
                    d = abs(ref - int(f[0:8]))
                    if (d < diff):
                        diff = d
                        filename = f
        except IOError:
            logger.error('Unable to find data files in "%s"', path)
        else:
            if(filename == None):
                logger.error('Unable to find data files in "%s"', path)
            else:
                csvpath = os.path.join(path, filename)
                csv = self.read_file(csvpath, index_column=index_column, unix=False)
        
        return csv

    def write_file(self, path, data):
        filename = data.index[0].astimezone(self.timezone).strftime('%Y%m%d_%H%M%S') + '_sim.csv';
        filepath = os.path.join(path, filename)
        
        data.index.name = 'time'
        data.tz_convert(self.timezone).astype(float).round(3).to_csv(filepath, sep=self.separator, decimal=self.decimal, encoding='utf-8')

    def concat_file(self, path, data):
#         filename = data.index[0].astimezone(self.timezone).strftime('%Y') + '_sim.csv'
        filename = data.index[0].strftime('%Y') + '_sim.csv'
        filepath = os.path.join(path, filename)
        
        data.index.name = 'time'
        if os.path.isfile(filepath):
            csv = pd.read_csv(filepath, sep=self.separator, decimal=self.decimal, index_col='time', parse_dates=['time'])
#             csv.index = csv.index.tz_localize(tz.utc).tz_convert(self.timezone)
        else:
            csv = pd.DataFrame()
        
        # Concatenate data to existing file
        # Preserve column order, as pandas concatenation may sort them as result of a bug (https://github.com/pandas-dev/pandas/issues/4588)
#         csv = pd.concat([csv, data.tz_convert(self.timezone)])
        csv = pd.concat([csv, data])
        csv[data.columns].astype(float).round(3).to_csv(filepath, sep=self.separator, decimal=self.decimal, encoding='utf-8')

