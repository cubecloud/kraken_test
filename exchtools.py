import os
import re
import time
import datetime
import pytz
import ccxt
import pandas as pd
from tqdm import tqdm
from flask import Flask, jsonify , make_response
from flask_restful import Api, Resource
from typing import Tuple
from secure import Secure
from dateutil.rrule import rrule, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY
from sqlalchemy import Column, Integer, String, Float, create_engine, ForeignKey, Table, DateTime, BigInteger


from dotenv import load_dotenv, find_dotenv

__version__ = 0.00017


app = Flask(__name__)
api = Api(app)


class CryptoAPI(Resource):
    def __init__(self, **kwargs):
        self.exchtools = kwargs['exchange']
        self.refined_pairs = [re.sub('\W+','', pair) for pair in self.exchtools.pairs_symbols]
        pass

    def get(self, symbol: str):
        if symbol.upper() in self.refined_pairs:
            idx = self.refined_pairs.index(symbol.upper())
            results = self.exchtools.get_day_high_low(self.exchtools.pairs_symbols[idx])
        elif symbol == "refresh":
            self.exchtools.refresh_data()
            return make_response(jsonify({'Ok.': f'{self.exchtools.pairs_symbols} refreshed'}), 200)
        else:
            return self.page_not_found()
        return make_response(jsonify(results), 200)

    @app.errorhandler(404)
    def page_not_found(self):
        return make_response(jsonify({'error': 'Not found'}), 404)


class ExchTools:
    def __init__(self, pairs_symbols, timeframes):
        self.key, self.secret = 'YOUR_API_KEY', 'YOUR_SECRET'
        self.exchange_id: str = 'kraken'
        self.exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = self.exchange_class({'apiKey': self.key, 'secret': self.secret})
        self.timezone = pytz.timezone(self.get_local_timezone_name())
        self.datetime_format: str = "%Y-%m-%d %H:%M:%S"
        self.ohlcv_cols: Tuple = ('datetime',
                                  'open',
                                  'high',
                                  'low',
                                  'close',
                                  'volume',
                                  )
        self.ohlcv_dtypes = {'datetime': DateTime,
                             'open': Float,
                             'high': Float,
                             'low': Float,
                             'close': Float,
                             'volume': Float,
                             }
        self.pairs_symbols: list = pairs_symbols
        self.timeframes: list = timeframes
        self.database: str = "orm.sqlite"
        self.fullpath_db: str = f'sqlite:///{self.database}'
        self.engine = create_engine('sqlite:///orm.sqlite', echo=False)

        if os.path.isfile(os.path.join(os.getcwd(), self.database)):
            self.db_base_exist: bool = True
        else:
            self.db_base_exist: bool = False
        self.refresh_data()
        pass


    @staticmethod
    def get_local_timezone_name():
        if time.daylight:
            offset_hour = time.altzone / 3600
        else:
            offset_hour = time.timezone / 3600

        offset_hour_msg = f"{offset_hour:.0f}"
        if offset_hour > 0:
            offset_hour_msg = f"+{offset_hour:.0f}"
        return f'Etc/GMT{offset_hour_msg}'

    def set_api_keys(self) -> None:
        """
        Secure api keys with user SALT. 32 symbols total, crypting the keys, and add to .env file
        """
        load_dotenv(find_dotenv())
        secure_key = Secure()
        self.key, self.secret = secure_key.get_key(os.path.join(os.getcwd(), ".env"))
        self.exchange_id = 'kraken'
        self.exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = self.exchange_class({'apiKey': self.key, 'secret': self.secret})
        pass

    @staticmethod
    def get_x_months_ago(months_qty, start_date=None):
        """
        Calculate datetime for x months ago
        Args:
            months_qty (int):       months ago qty
            start_date (datetime):  optional not provided used datetime.datetime.today

        Returns:
            datetime (object):      calculated object
        """
        if start_date is not None:
            today = start_date
        else:
            today = datetime.datetime.today()
        years = months_qty // 12
        month = months_qty % 12
        if today.month <= months_qty % 12:
            if month == today.month:
                month_calc = 12
                years += 1
            else:
                month_calc = today.month - month
                if years == 0:
                    years = 1
            x_months_ago = today.replace(year=today.year - years, month=month_calc)
        else:
            extra_days = 0
            while True:
                try:
                    x_months_ago = today.replace(year=today.year - years,
                                                 month=today.month - month,
                                                 day=today.day - extra_days)
                    break
                except ValueError:
                    extra_days += 1
        return x_months_ago

    def __prepare_limit(self, start_date, until_date, timeframe='1h') -> Tuple[int, list]:
        """
        Returns the qty of timeframes from start_date until until_date

        Args:
            start_date (str):   start_date in %Y-%m-%d %H:%M:%S format
            until_date (str):   until_date in %Y-%m-%d %H:%M:%S format
            timeframe (str):    timeframe

        Returns:
            timeframe_qty (int):    qty  of timeframes in time range
            frames_list (list):     list with datetime objects for each frame
        """
        timeframe_dict = {'1m': (MINUTELY, 1),
                          '3m': (MINUTELY, 3),
                          '5m': (MINUTELY, 5),
                          "15m": (MINUTELY, 15),
                          '30m': (MINUTELY, 30),
                          '1h': (HOURLY, 1),
                          '2h': (HOURLY, 2),
                          '4h': (HOURLY, 4),
                          '6h': (HOURLY, 6),
                          '8h': (HOURLY, 8),
                          '12h': (HOURLY, 12),
                          '1d': (DAILY, 1),
                          '3d': (DAILY, 3),
                          '1W': (WEEKLY, 1),
                          '1M': (MONTHLY, 1),
                           }

        if start_date == 'undefined':
            start_date = self.get_x_months_ago(1)
        else:
            start_date = datetime.datetime.strptime(start_date, self.datetime_format)

        if until_date == 'undefined':
            until_date = datetime.datetime.utcnow()
        else:
            until_date = datetime.datetime.strptime(until_date, self.datetime_format)

        timeframe_freq, timeframe_interval = timeframe_dict.get(timeframe, (HOURLY, 1))

        frames_list = list(rrule(dtstart=start_date,
                                 until=until_date,
                                 freq=timeframe_freq,
                                 interval=timeframe_interval))
        timeframe_qty = len(frames_list)
        return timeframe_qty, frames_list

    def __get_chunk_OHLCV(self, symbol, timeframe, since, limit):
        chunk = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        return chunk

    def get_OHLCV(self, symbol, timeframe='1h', since_datetime='undefined', until_datetime='undefined') -> list:
        """
        Args:
            symbol (str):           pair_symbol
            timeframe (str):        timeframe
            since_datetime (str):   start_date in %Y-%m-%d %H:%M:%S format
            until_datetime (str):   until_date in %Y-%m-%d %H:%M:%S format
        Returns:
            data (list):            OHLCV data in list format

            exchange 'since' examples:
            # exchange.parse8601 ('2018-01-01T00:00:00Z') == 1514764800000 // integer, Z = UTC
            # exchange.iso8601 (1514764800000) == '2018-01-01T00:00:00Z'   // iso8601 string
            # exchange.seconds ()      // integer UTC timestamp in seconds
            # exchange.milliseconds () // integer UTC timestamp in milliseconds
        """

        timeframes_qty, _ = self.__prepare_limit(start_date=since_datetime,
                                                 until_date=until_datetime,
                                                 timeframe=timeframe)

        since = self.exchange.parse8601(self.get_x_months_ago(1).strftime(self.datetime_format))
        if since_datetime != 'undefined':
            since = self.exchange.parse8601(since_datetime)
        _data = self.__get_chunk_OHLCV(symbol, timeframe, since, limit=timeframes_qty)
        return _data


    @staticmethod
    def __refine_symbol(symbol):
        """
        Args:
            symbol (str):   symbol w/ special characters (BTC/USD, etc)

        Returns:
            refined_symbol (str):   remove special characters
        """
        refined_symbol = re.sub('\W+','', symbol)
        return refined_symbol

    def ohlcv_to_sql(self, symbol, timeframe, start_datetime='undefined', until_datetime='undefined'):
        _data = self.get_OHLCV(symbol,
                               timeframe=timeframe,
                               since_datetime=start_datetime,
                               until_datetime=until_datetime
                               )

        data_df = pd.DataFrame(_data,
                               columns=self.ohlcv_cols
                               )
        data_df['datetime'] = pd.to_datetime(data_df['datetime'], unit='ms')
        _symbol = self.__refine_symbol(symbol)
        data_df.to_sql(name=f'{_symbol}_{timeframe}_ohlcv',
                       con=self.engine,
                       index=False,
                       if_exists='append',
                       dtype=self.ohlcv_dtypes
                       )
        pass

    def __get_bad_idxs_OHLCV(self,
                             bad_idxs: list,
                             symbol: str,
                             timeframe: str = '1h'
                             ) -> list:
        lost_data = []
        _idx = pd.DataFrame(index=bad_idxs)
        if pd.Index(_idx.index).is_monotonic_increasing:
            _limit = len(bad_idxs)
            bad_idxs = [bad_idxs[0]]
        else:
            _limit = 1
        for since_idx in bad_idxs:
            since = self.exchange.parse8601(since_idx.strftime(self.datetime_format))
            _data = self.__get_chunk_OHLCV(symbol, timeframe, since, limit=_limit)
            lost_data.extend(_data)
        return lost_data

    def check_and_load(self, symbol: str, timeframe='1h', start_datetime=None, end_datetime=None):
        def _clean_dupes(df):
            df = df.drop_duplicates(keep='first')
            return df

        def get_bad_idxs(ctrl_df, df):
            _df = pd.concat([ctrl_df, df], axis=1)
            _mask = _df['datetime'].isna()
            _indexes = _df[_mask].index.to_list()
            _indexes = [idx.to_pydatetime() for idx in _indexes]
            return _indexes

        def add_lost_data(df, _lost_data):
            datetime_int = df.datetime.astype('int64') // 10 ** 6
            df = df.drop(columns=['datetime'])
            df.reset_index(drop=True)
            df.insert(loc=0, column='datetime', value=datetime_int)
            add_df = pd.DataFrame(data=lost_data,
                                  columns=self.ohlcv_cols,
                                  )
            add_df_index = pd.to_datetime(add_df['datetime'], unit='ms')
            add_df.index, add_df.index.name = add_df_index, 'datetimeindex'
            df = df.append(add_df, ignore_index=True)
            df = df.sort_values(by='datetime').reset_index(drop=True)
            return df

        _symbol = self.__refine_symbol(symbol)
        data_df = pd.read_sql_table(table_name=f'{_symbol}_{timeframe}_ohlcv',
                                    con=self.engine,
                                    columns=self.ohlcv_cols,
                                    parse_dates={'datetime': self.datetime_format}
                                    )
        if start_datetime is None:
            start_datetime = data_df.iloc[:1, 0].values[0]
            sd_ts = pd.to_datetime(start_datetime, unit='ms')
            start_datetime = sd_ts.to_pydatetime().strftime(self.datetime_format)
        if end_datetime is None:
            end_datetime = data_df.iloc[-1:, 0].values[0]
            ed_ts = pd.to_datetime(end_datetime, unit='ms')
            end_datetime = ed_ts.to_pydatetime().strftime(self.datetime_format)

        data_df.index, data_df.index.name = data_df['datetime'], 'datetimeindex'
        """ Creating control pd.Series with True datetimeindex """
        interval = pd.date_range(start_datetime, end_datetime, freq='H')
        control_df = pd.DataFrame()
        control_df.index = interval

        if data_df.shape[0] == interval.shape[0]:
            """ Checking for monotonic datetime """
            if not pd.Index(data_df.index).is_monotonic_increasing:
                new_df = _clean_dupes(data_df)
                new_df = new_df.sort_values(by='datetime').reset_index(drop=True)
                new_df.index = new_df['datetime']
                indexes = get_bad_idxs(control_df, new_df)
                if indexes:
                    print(f'Missed indexes qty = {len(indexes)}')
                    lost_data = self.__get_bad_idxs_OHLCV(indexes, symbol, timeframe)
                    new_df = add_lost_data(new_df, lost_data)
                """ Saving reconstructed data """
                new_df.to_sql(name=f'{_symbol}_{timeframe}_ohlcv',
                              con=self.engine,
                              index=False,
                              if_exists='replace',
                              dtype=self.ohlcv_dtypes,
                              )
        else:
            new_df = _clean_dupes(data_df)
            new_df = new_df.sort_values(by='datetime').reset_index(drop=True)
            new_df.index = new_df['datetime']

            """ begin for testing purpose """
            # new_df = new_df.iloc[40:, :]
            """ end for testing purpose """

            indexes = get_bad_idxs(control_df, new_df)
            if indexes:
                print(f'Missed indexes qty = {len(indexes)}')
                lost_data = self.__get_bad_idxs_OHLCV(indexes, symbol, timeframe)
                new_df = add_lost_data(new_df, lost_data)

            """ Saving reconstructed data """
            new_df.to_sql(name=f'{_symbol}_{timeframe}_ohlcv',
                          con=self.engine,
                          index=False,
                          if_exists='replace',
                          dtype=self.ohlcv_dtypes,
                          )

        """ Everything is ok. We can get new data """
        start_date = pd.to_datetime(str(end_datetime))
        timeframes_qty, frames_list = self.__prepare_limit(start_date=start_date.strftime(self.datetime_format),
                                                           until_date="undefined",
                                                           timeframe=timeframe
                                                           )
        """ Checking timeframes_qty """
        if timeframes_qty > 1:
            print(f'{symbol} loading timeframes qty: {timeframes_qty}')
            start_date = frames_list[1].strftime(self.datetime_format)
            since = self.exchange.parse8601(start_date)
            _data = self.__get_chunk_OHLCV(symbol, timeframe, since, limit=timeframes_qty)
            add_df = pd.DataFrame(data=_data,
                                  columns=self.ohlcv_cols,
                                  )
            add_df_index = pd.to_datetime(add_df['datetime'], unit='ms')
            add_df = add_df.drop(columns=['datetime'])
            add_df.insert(loc=0, column='datetime', value=add_df_index)
            add_df.index, add_df.index.name = add_df_index, 'datetimeindex'
            add_df.to_sql(name=f'{_symbol}_{timeframe}_ohlcv',
                          con=self.engine,
                          index=False,
                          if_exists='append',
                          dtype=self.ohlcv_dtypes
                          )
        pass

    def refresh_symbol(self, symbol, timeframe):
        if self.db_base_exist:
            self.check_and_load(symbol, timeframe)
        else:
            self.ohlcv_to_sql(symbol, timeframe)
        pass

    def refresh_data(self):
        start_time = datetime.datetime.now(self.timezone)
        print(f"\nChecking pairs: {self.pairs_symbols}\nwith timeframes: {self.timeframes}")
        msg = f"Total qty of Kraken symbols: {len(self.pairs_symbols)} * {len(self.timeframes)} " \
              f"= {len(self.pairs_symbols) * len(self.timeframes)}"
        print(msg)
        print(f"Starting at: {start_time}")

        # progress_bar = tqdm(self.pairs_symbols)

        # for symbol in progress_bar:
        for symbol in self.pairs_symbols:
            for timeframe in self.timeframes:
                # progress_bar.set_description(f"Processing: {symbol} {timeframe}...")
                if self.db_base_exist:
                    self.check_and_load(symbol, timeframe)
                else:
                    self.ohlcv_to_sql(symbol, timeframe)

        self.db_base_exist = True
        end_time = datetime.datetime.now(self.timezone)
        print(f'Downloaded data for pairs: {self.pairs_symbols} with intervals: {self.timeframes}\n'
              f'Total time elapsed: {end_time - start_time}')
        pass

    def get_day_high_low(self, symbol, timeframe='1h'):
        _symbol = self.__refine_symbol(symbol)
        data_df = pd.read_sql_table(table_name=f'{_symbol}_{timeframe}_ohlcv',
                                    con=self.engine,
                                    columns=self.ohlcv_cols,
                                    parse_dates={'datetime': self.datetime_format}
                                    )
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
        data_df = data_df.set_index("datetime")
        data_high_max = data_df.loc[data_df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 1]]
        data_low_min = data_df.loc[data_df.groupby(pd.Grouper(freq='D')).idxmin().iloc[:, 2]]

        col_time1 = data_low_min.index.strftime(self.datetime_format)
        data_low_min = data_low_min.reset_index(drop=True)
        data_low_min.insert(loc=0, column='time', value=col_time1)
        data_low_min.insert(loc=0, column='type', value="min")

        col_time = data_high_max.index.strftime(self.datetime_format)
        data_high_max = data_high_max.reset_index(drop=True)
        data_high_max.insert(loc=0, column='time', value=col_time)
        data_high_max.insert(loc=0, column='type', value="max")

        result = []
        for idx in range(1, data_low_min.shape[0]):
            result.append(data_low_min[idx - 1:idx].to_json(orient="records"))
            result.append(data_high_max[idx - 1:idx].to_json(orient="records"))

        return result


if __name__ == '__main__':
    pairs_symbols = ['BTC/USD',
                     'ETH/USD',
                     'XRP/EUR',
                     'XRP/USD',
                     ]
    frames = ['1h']
    exchtools = ExchTools(pairs_symbols, frames)
    api.add_resource(CryptoAPI, "/<string:symbol>", resource_class_kwargs={'exchange': exchtools})
    app.run(debug=True)





