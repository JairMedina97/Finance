#Python 2.7

#Make necessary imports
import datetime
import datetime as dt
import pandas_datareader as pdr
import numpy as np

#set the time frame to fetch stock data
start = dt.datetime(2016,1,1)
end = dt.datetime(2019, 11, 11)

#choose the stock you want to get it's info for
ticker   = input("Enter stock ticker : ")

#request the data
df = pdr.get_data_yahoo(ticker, 
                          start=datetime.datetime(2008, 1, 1), 
                          end=datetime.datetime(2019, 9, 2))

def append_change_column(self, df, ticker):
        """
         take the dataframe that holds the stock info and the ticker
         of interest then append the new change column and the
         new close column to the main dataframe
        """
        df2 = pd.DataFrame()
        df2['change'] = np.log(df['close']) - np.log(df['close'].shift(1))
        self.main_df[str(ticker) + 'CHG'] = df2['change']
        self.main_df[str(ticker) + 'CLS'] = df['close']

        return self.main_df

    

def create_slope_sum_market(df):
    """
    Takes a dataframe and looks at all the columns with perfecnt change. Then it compairs the stock of intreste and caculates
    the differnce between it and the rest of the percent changes on that day in the market. Finally it sums them up.
    returns the oringal dataframe with the new columns called Slope_sum
    """

    columns = df.columns

    # filter the columns with CHG vs CLS
    CLS_columns, CHG_columns = get_columns_with_CLS(columns)

    # print CHG_columns, ' chagne columns'

    column_index = 1

    while column_index < len(CHG_columns) - 1:

        stock_looking_at = CHG_columns[column_index]

        # make dataframe of a bunch of zeros to the size of the df coming in
        slope_sum = pd.DataFrame(np.zeros((len(df.index), 1)))

        for stock in CHG_columns:

            if stock != stock_looking_at:

                slope_sum[0] = slope_sum[0] + \
                    df[CHG_columns[column_index]] - df[stock]

        # add the newly formed column containing the slope info to the main df
        df[str(CHG_columns[column_index].replace('CHG', 'slope_sum'))] = slope_sum

        column_index += 1
    # print df
    return df



def generate_target_values(df, batch_count, column_name, look_ahead):
    """
    Takes a dataframe and a batch count to gernate the % change values good or bad for each batch
    """

    list_of_closes = df[column_name].tolist()

    i = 0
    target_values = []
    while i < (len(list_of_closes) - (look_ahead + batch_count) + 1):

        target_day_index = i + batch_count + look_ahead - 1

        percent_change = find_percent_change(
            list_of_closes[target_day_index], list_of_closes[i + batch_count - 1])

        if percent_change < 0:
            target_values.append(-1)
        else:
            target_values.append(1)
        i += 1
    number_of_target_values = len(target_values)

    return target_values, number_of_target_values

class Support_Vector():
    """
    class used to handle the ML
    """ 
  def __init__(self, X, Y):
        self.X = X
        self.Y = Y    
  
  def train(self, keep=False):
        """
        x and y are the features and the target values for the differnt pairs that we want the model to learn over
        """
        #split the features and target values 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.2)

        svclassifier = SVC()

        svclassifier.fit(self.X_train, self.y_train)

        self.y_pred = svclassifier.predict(self.X_test)

        #save the model in the object
        self.model = pickle.dumps(svclassifier)

        #  Saves the model to disk
        if keep:
            joblib.dump(svclassifier, 'models/model' +
                        str(datetime.now()) + '.pkl')




  def generate_buy_sells(self, batch):
        """
        This function takes batches of slope sums and applies that to the model it returns 1 or -1
        """
        return self.model.predict([batch])

    def append_list_of_buy_sells(self, list_of_batches, column_name):
        """
        Takes a list of batches and makes a new column in the self.main_df
        that will store the differnt values -1/1
        """
        array_of_buy_sells = []
        for batch in list_of_batches:
            should_buy_or_sell = self.generate_buy_sells(batch)
            # print should_buy_or_sell, ' should buy sell'
            array_of_buy_sells.append(int(should_buy_or_sell))


        array_of_nones = []
        for i in range(len(self.main_df) - len(array_of_buy_sells)):
            array_of_nones.append(None)

            # NOTE
            # needed to add the nones to the begining of the buy_sell array since
            # it starts calculating whether or not to buy or sell a few days
            # into the close values

        # remove the last bid
        del array_of_buy_sells[-1]
        array_of_nones.append(None)

        self.main_df[column_name.replace(
            'slope_sum', 'bid_stream')] = array_of_nones + array_of_buy_sells

        return self.main_df
            
