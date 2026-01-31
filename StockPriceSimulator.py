import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import matplotlib.dates as mpl_dates
import mplfinance as mpf
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde, iqr
from collections import OrderedDict
from matplotlib.gridspec import GridSpec


class simulateStockData:
    def __init__(self, ticker:str, period:int, interval:str) -> None:
        """Initializes a Stock price simulator

        Args:
            ticker (str): yahoo finance ticker of desired stock
            period (int): the time period in terms of days
            interval (str): the intervals of the data sampling (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)
        """
        # Asserting proper interval format
        try:
            assert(interval in ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"])
        except:
            raise ValueError('Given interval format not valid')
        
        # Setting inital attributes
        self.ticker = ticker # Setting the stock ticker
        self.period = period
        self.interval = interval
        
        # Specific filename for the data given the inputs
        self.filename = f"{self.ticker}_stock_{self.period}_period_{self.interval}_intervals.csv"
        
        # Downloading the stock data if not in the folder
        if self.filename not in os.listdir(os.getcwd()):
            self.__getStockData()
        
        # Loading a pandas dataframe of the historical data
        self.tickerData = pd.read_csv(self.filename)
        
        # Formating the data given the file
        self.__formatData()
        
        # Getting the KDE of the whole Data:
        self.__wholeKDE = self.__getKDE(self.tickerData["Open_diff"].dropna(), "SilvermanUni")
        
        
        
        
    def __getStockData(self) -> None:
        """Gets the stock historical data of the ticker using yahoo finance library and adds an attribute. Always assumes starting from today
        """
        
        # Setting up the time period using datetime library, assuming starting today
        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.period) 
        
        # Getting the data downloaded
        data = yf.download(
            self.ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust= True, # Adjusting price to uniform prices
            progress= True
        )
        
        data.to_csv(self.filename) # Specific file name with the ticker, period, and intervals in it

        
        
    def __formatData(self) -> None:
        """takes the data file and formats it into a useable data format for processing, specifically setting data types and removing bad rows 
        """
        # Modifying data to make sure that it is formatted correctly 
        self.tickerData = self.tickerData[2:].reset_index(drop=True) # Dropping first two rows and reindexing
        self.tickerData = self.tickerData.rename({'Price' : 'Date'}, axis=1) # Renaming columns to proper names
        self.tickerData[["Close", "High", "Low", "Open"]] = self.tickerData[["Close", "High", "Low", "Open"]].astype(float) # Getting the datatype to be able to be read correctly
        self.tickerData["Date"] = pd.to_datetime(self.tickerData["Date"]) # Setting the date column to a datetime object
        
        # Creating a daily open differential column
        self.tickerData["Open_diff"] = self.tickerData['Open'] - self.tickerData["Open"].shift(1)

        
        
    def reindexDataframe(self, df:pd.DataFrame, date_freq:str, agg_method:str) -> pd.DataFrame:
        """Resamples a time-based dataframe and uses an aggregation function in the resample

        Args:
            df (pd.DataFrame): Dataframe to resample timeline and apply aggregation function
            date_freq (str): Frequency of resampling ('w', 'm', 'y')
            agg_method (str): Method of aggregation of resampling (mean, total, median)

        Raises:
            ValueError: Wrong date frequency format used
            ValueError: Wrong aggregation function format used
            KeyError: A "Date" column not in the supplied dataframe

        Returns:
            pd.DataFrame: Resampled and aggregated dataframe
        """
        
        # Keys to method functions
        freq_map = {
            'w': 'W',
            'm': 'ME', # Month end
            'y': 'Y',
        }

        agg_map = {
            'mean': 'mean',
            'total': 'sum',
            'median': 'median'
        }

        # Raising errors if inputs are not correct
        if date_freq not in freq_map:
            raise ValueError("date_freq must be 'w', 'm', or 'y'")

        if agg_method not in agg_map:
            raise ValueError("agg_method must be 'mean', 'total', or 'median'")
        
        if "Date" not in df.columns:
            raise KeyError("DataFrame must contain a 'Date' column")

        # Returns resampled data
        return (
            df.set_index("Date")
            .resample(freq_map[date_freq]) # Resampling data
            .agg(agg_map[agg_method]) # Aggregation function
            .reset_index()
        )

    
    
    def __createQuartiles(self, column = "Open_diff") -> pd.DataFrame:
        """Breaks a desired column of stock data into quartiles and returns a dataframe contianing the column and their quartiles

        Args:
            column (str, optional): Column of tickerData to break into quartiles. Defaults to "Open_diff".

        Returns:
            pd.DataFrame: A dataframe containing the non-NaN column of the ticker data and a column of their Quartiles as well as their next jump
        """
        
        # Dataframe of the desired column
        df = pd.DataFrame(self.tickerData[column])
        
        # Setting bins for the quartiles
        q25, q50, q75 = np.percentile(df[column].dropna().to_numpy(), [25, 50, 75])
        self.quartileBins = [-np.inf, q25, q50, q75, np.inf]
        
        # Adding quartile functionality
        df["quartile"] = pd.qcut(df[column], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        
        # Adding the next jump to the data
        df["next_jump"] = df[column].shift(-1)
        
        # Dropping NaN and returning
        return df.dropna()
        
    
    
    def plotStochasticPath(self, plotTitle:str, **kwargs) -> None:
        """Creates a plot of the stochastic path of the stock ticker or other data with the bottom of the plot being the bar chart of daily jumps

        Args:
            plotTitle (str): Desired title of the plot

        Raises:
            KeyError: Date not included as a column of the passed through dataframe 
        """
        
        # Default functionality 
        defaults = {
            "data" : self.tickerData,
            "show_ax2" : True,
            "ax1_column" : "Open",
            "ax2_column" : "Open_diff",
            "ax1_yLabel" : "Open Price",
            "ax2_yLabel" : "Prev Open - Today Open",
        }
        
        # Updating depending on kwargs passed through
        defaults.update(kwargs)
        
        # Checking for a Date column
        if "Date" not in defaults["data"].columns:
            raise KeyError("DataFrame must contain a 'Date' column")
        
        # Creatng figure and axes
        if defaults["show_ax2"]: # 2 plot functionality
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(14, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]}
            )
        else: # One plot functionality
            fig, ax1 = plt.subplots(
                1, 1,
                figsize=(14, 6)
            )
            ax2 = None

        # Top Axis Plotting
        ax1.plot(
            defaults["data"]["Date"],
            defaults["data"][defaults["ax1_column"]],
            linewidth=1.5
        )

        # Styling the top axis
        ax1.set_title(plotTitle, fontname="Franklin Gothic Medium", fontsize=18, pad=15)
        ax1.set_ylabel(defaults["ax1_yLabel"])
        ax1.grid(True)

        # Plotting the bottom axis (Optional)
        if ax2 is not None:
            # Getting the plot height limits
            ylimit = (
                defaults["data"][defaults["ax2_column"]].min(),
                defaults["data"][defaults["ax2_column"]].max()
            )

            # Plotting the second axis
            ax2.bar(
                defaults["data"]["Date"],
                defaults["data"][defaults["ax2_column"]],
                width=2,      # adjust if dates overlap
                alpha=0.8
            )
            
            # Reduce x-axis clutter (important for 30y of data)
            #ax2.xaxis.set_major_locator(plt.MaxNLocator(10))

            # Styling the second axis
            ax2.set_ylim(ylimit)
            ax2.set_ylabel(defaults["ax2_yLabel"])
            ax2.set_xlabel("Date")
            ax2.grid(False)

        # Layout
        plt.tight_layout()
        plt.show()
    
    
    
    def plotCandlestick(self, plotTitle:str, resamplePeriod:str = None, plotVolume:bool = True, movingAvg:tuple = (7, 30)) -> None:
        """Plots a candlestick graph of the class stock ticker

        Args:
            plotTitle (str): Title of the plot
            resamplePeriod (str, optional): Period if the data time period is to be resampled (Resamples by mean). Defaults to None.
            plotVolume (bool, optional): Boolean to decide wether or not to plot the volume at the bottom. Defaults to True.
            movingAvg (tuple, optional): tuple of moving integers that represent desired moving averages to be included in graph. Defaults to (7, 30).

        Raises:
            TypeError: movingAvg variable not a tuple
            TypeError: movingAvg variable does not contain only integers
        """
        
        # Checking proper format of the moving average variable
        if not isinstance(movingAvg, tuple):
            raise TypeError("Moving average must be a tuple")
            
        if not all(type(i) is int for i in movingAvg):
            raise TypeError("Moving average must be a with only integers")
            
        # Reformatting the dataframe
        df = self.tickerData[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
        df.loc[:, 'Volume'] = df['Volume'].astype(object).apply(int)
        
        # Resampling data if asked for
        if resamplePeriod != None:
            df = self.reindexDataframe(df, resamplePeriod, 'mean')
            
        # Setting the index to be the date
        df = df.set_index('Date')
        
        # Setting the arguments for the plot
        kwargs = dict(
            type = 'candle',
            style = 'yahoo',
            volume=plotVolume,
            mav=movingAvg,
            figscale=2,
            figratio=(18,10),
            title=plotTitle,
            returnfig=True
        )
        
        # Plotting the camdlestick
        fig, axes = mpf.plot(df, update_width_config=dict(candle_linewidth=0.4), **kwargs)
        
        # Adding legend for moving averages
        ax = axes[0]  # main price axis
        handles = ax.get_lines()
        labels = [f"MA {m}" for m in movingAvg]
        ax.legend(handles, labels, loc='upper left')
        
            
    
    def createBins(self, values:pd.Series, binSize:int) -> np.ndarray:
        """Takes data from a column and returns the array of bins for a histogram with desired size

        Args:
            values (pd.Series): the data column values to be binned
            binSize (int): integer of the desired size between each bin

        Returns:
            np.ndarray: an ndarray of the bins
        """
        
        # Calculating the endges of the bins
        min_val = np.floor(values.min() / binSize) * binSize
        max_val = np.ceil(values.max() / binSize) * binSize

        # Getting bins
        bins = np.arange(min_val, max_val + binSize, binSize)
        
        return bins
    
    
    
    def __calcultateSilvermansUniRule(self, values:pd.Series) -> float:
        """Takes in observed values and produces a badwidth factor using Silverman\'s rule of thumb to be used for a kde 

        Args:
            values (pd.Series): Pandas series of the observed values of the distribution to be estimated 

        Returns:
            float: Calculated bandwidth factor
        """
        
        values = np.asarray(values)
        n = len(values)

        # Silverman's univariate rule of thumb
        sigma = np.std(values, ddof=1)
        iqr_val = iqr(values)
        sigma_hat = min(sigma, (iqr_val / 1.34))

        h = 1.06 * sigma_hat * n ** (-1 / 5)

        # Convert bandwidth to gaussian_kde scaling factor
        bw_factor = h / np.std(values, ddof=1)
        
        return bw_factor

        
        
    def __getKDE(self, data:pd.Series, method:str) -> gaussian_kde:
        """Takes in the data and a bandwidth method to return a gaussian kde

        Args:
            data (pd.Series): Data to extract a distribution from
            method (str): banwidth method, typically Silvermans rule of thumb

        Returns:
            scipy.stats.gaussian_kde: kde object to be used 
        """
        
        # Creating 
        if method == "SilvermanUni":
            bw_factor = self.__calcultateSilvermansUniRule(data)
        else: 
            bw_factor = 1 # Placeholder
        
        # Getting the values as a np array
        values = np.asarray(data.dropna())
        
        #print(values)
        # Getting the KDE
        kde = gaussian_kde(values, bw_method=bw_factor)

        return kde
    
    
    
    def plotDistribution(self, plotTitle: str, probModel:bool = False, **kwargs) -> None:
        """Creates a histogram based on occurances or probability with options to include a KDE and different bin sizes 

        Args:
            plotTitle (str): Title for the plot
            probModel (bool, optional): Show histogram in terms of probability instead of counts. Defaults to False.

        Raises:
            ValueError: raises when getKDE is true and probModel is not true
        """
        
        # Default functionality
        defaults = {
            "data"   : self.tickerData,
            "column" : "Open_diff",
            "xLabel" : "Previous Day Open − Today Open",
            "yLabel" : "Occurances",
            "getKDE" : False,
            "KDEmethod" : "SilvermanUni",
            "binSize" : 5,
            "logScale" : False,
            "returnPlot" : False,
            "OtherAxis" : None,
            "uniformBinPlot" : False,
            "yLimit" : None
        }
        
        # Updating defaults with kwargs
        defaults.update(kwargs)
        
        # Getting values
        values = defaults["data"][defaults["column"]].dropna()
        
        # Asserting that getKDE is only true when probModel is True
        if kwargs.get("getKDE", False) and not probModel:
            raise ValueError("getKDE can only be True when probModel is True")
        
        # Getting the bins from the method
        if defaults["uniformBinPlot"] == True:
            bins = self.createBins(self.tickerData["Open_diff"].dropna(), defaults["binSize"])
        else:
            bins = self.createBins(values, defaults["binSize"])
        
        # Plot histogram and capture outputs
        if defaults["returnPlot"] == True and defaults["OtherAxis"] != None:
           ax = defaults["OtherAxis"]
        else:
            fig, ax = plt.subplots(figsize=(15, 9))

        counts, bin_edges, patches = ax.hist(values, bins=bins, density=probModel, alpha = 0.6) # Density makes it a probibailistic model

        # Offset Styling factors
        x_offset = 0.0
        y_offset_factor = 1.01

        # Annotate counts above bars if not a probibalistic model
        if probModel == False:
            for count, patch in zip(counts, patches):
                if count > 0:  # avoid log(0) issues
                    x = patch.get_x() + patch.get_width() / 2
                    y = patch.get_height()
                    
                    # Shift left/right based on sign of bin center
                    bin_center = x
                    if bin_center < 0:
                        x_adj = x - patch.get_width() * 0.2  # shift left
                    else:
                        x_adj = x + patch.get_width() * 0.2   # shift right

                    # Shift upward slightly
                    y_adj = y * y_offset_factor

                    # Adding text
                    ax.text(
                        x_adj,
                        y_adj,
                        f"{int(count)}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation = 50
                    )
        
        # Adding kde if asked for
        if defaults["getKDE"] == True:
            kde = self.__getKDE(values, defaults["KDEmethod"])
            
            # x range for smooth curve
            x = np.linspace(bins.min(), bins.max(), 500)

            # Scale KDE to match histogram counts, uncomment when you take off density in ax.hist, LEFTOVER FUNCTIONALITY
            #kde_counts = kde(x) * len(values.dropna()) * defaults["binSize"]

            # Plot KDE
            ax.plot(
                x,
                kde(x),
                linewidth=2,
                label="Gaussian KDE"
            )

            ax.legend()
        
        # Customizing the styling of the plots
        if defaults["logScale"] == True:
            plt.yscale("log")
        
        # Setting ylimit for unifromity among multiple
        if defaults['yLimit'] != None:
            ax.set_ylim(top=defaults["yLimit"])
        
        ax.set_xlabel(defaults["xLabel"])
        ax.set_ylabel(defaults["yLabel"])
        
        if defaults["returnPlot"] == True:
            ax.set_title(plotTitle, fontname="Franklin Gothic Medium", fontsize=11, pad=7)

        if defaults["returnPlot"] == False:
            plt.tight_layout()
            plt.title(plotTitle, fontname="Franklin Gothic Medium", fontsize=18, pad=15)
            plt.show()

    
    
    def getQuartileDistributions(self, column = "Open_diff") -> None:
        """Gets the seperate dataFrames of each quartiles next jumps and puts them in a dictionary as well as puts their KDEs into another dictionary depending on quartile

        Args:
            column (str, optional): Column to get the quartiles of, not recommended to do anything but Open_diff. Defaults to "Open_diff".
        """
        
        # Creating the quartiles
        df = self.__createQuartiles(column=column)
        
        # Creating four dataframes of all the jumps after a jump in the quartile
        next_jump_dfs = {
            q: df.loc[df["quartile"] == q, ["next_jump"]].dropna()
            for q in df["quartile"].unique()
        }
        
        # Order wanted for dictionary
        desired_order = ["Q1", "Q2", "Q3", "Q4"]

        # Reorders the dictionary of dataframes next jumps for each quartile
        self.next_jump_dfs_qt = OrderedDict(
            (key, next_jump_dfs[key]) for key in desired_order
        )
        
        # Getting the KDEs for the quartiles next jumps
        self.next_jump_KDEs = {
            q: self.__getKDE(df.loc[df["quartile"] == q]["next_jump"].dropna(), "SilvermanUni")
            for q in df["quartile"].unique()
        }
        
        return



    def plotQuartileDistributions(self) -> None:
        """Plots the jumps that happen after a jump that is in each quartile. In other words if we get a jump in a specific quartile, these are the distributions of the likely next jumps
        """
        
        # Seeing if getQuartile distributions has been called yet
        try:
            assert(hasattr(self, "next_jump_dfs_qt"))
        except:
            self.getQuartileDistributions()
        
        # Creating axis
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plotting each quartile plot histogram
        for ax, (subtitle, data) in zip(axes.flat, self.next_jump_dfs_qt.items()):
            self.plotDistribution(subtitle, 
                                  data=data,
                                  probModel=True, 
                                  OtherAxis = ax, 
                                  column = "next_jump", 
                                  returnPlot = True, 
                                  getKDE=True, 
                                  uniformBinPlot = True, 
                                  yLimit=0.05
                                  )

        # Displaying layout
        fig.suptitle("Quartile Distribution Jumps", fontname="Franklin Gothic Medium", fontsize=18)
        plt.tight_layout()
        plt.show()
        
        
        
    def returnQuartile(self, x:float) -> str:
        """Sorts a number into a quartile depending on the distribution of the stock daily price changes

        Args:
            x (float): number that is to be sorted into a quartile

        Returns:
            str: the quartile the number is found in (Q1, Q2, Q3, Q4)
        """
        
        # Checking the Quartile
        if x <= self.quartileBins[1]:
            return "Q1"
        elif x <= self.quartileBins[2]:
            return "Q2"
        elif x <= self.quartileBins[3]:
            return "Q3"
        else:
            return "Q4"
        
    
    
    def singleJump(self, lastJump: float, method:str = "whole") -> float:
        """Takes in the last jump that happened, finds the quartile of it, then samples from the conditional 
        distribution of all jumps that happened after a jump in the same quartile

        Args:
            lastJump (float): te float number of the last jump that happens
            method (str): method to simulate the path (whole, quartile) Either whole distribution or conditional quartile distributions.

        Returns:
            float: New jump sampled from the distribution of jumps that happen after the quartile of the last jump
        """
        # Seeing if a valid method is passed through
        try:
            assert(method in ["whole", "quartile"])
        except:
            ValueError("Invalid simulation sampling method used. Either use whole or quartile")
        
        # Quartile sampling
        if method == "quartile":
            # Getting the quartile of the last jump
            conditionalQuartile = self.returnQuartile(lastJump)
            
            # Getting the KDE of the jumps after the jumps in the same quartile
            samplingKDE = self.next_jump_KDEs[conditionalQuartile]
            
            
            # Sampling a new value from the conditional KDE
            newJump = samplingKDE.resample(1)
        
        # Whole distribution sampling
        elif method == "whole":
            newJump = self.__wholeKDE.resample(1)
        
        # Returning the new jump to do
        return newJump.item(0)
    
    
        
    def simulatePath(self, startingPrice:int, steps:int, method:str = "whole") -> list:
        """Simulates a fake path of the price history of the stock ticker of the class using the whole and conditional distributions of the actual data 

        Args:
            startingPrice (int): price to start the simulation at
            steps (int): number of price steps to take
            method (str): method to simulate the path (whole, quartile) Either whole distribution or conditional quartile distributions.

        Returns:
            list: list of simulated price history that mimics the distribution and path of the stock ticker of the class
        """
        
        # Seeing if getQuartile distributions has been called yet
        try:
            assert(hasattr(self, "next_jump_dfs_qt"))
        except:
            self.getQuartileDistributions()
        
        # Seeing if a valid method is passed through
        try:
            assert(method in ["whole", "quartile"])
        except:
            ValueError("Invalid simulation sampling method used. Either use whole or quartile")
        
        # List of the price history 
        priceHistory = [startingPrice]
        
        # Last jump
        lastJump = 0
        
        # For loop to go over all the price jumps
        for i in range(steps):
            
            # Base Case Condition for first jump
            if i == 0:
                # Getting a jump based off the whole distribution
                lastJump = self.__wholeKDE.resample(1).item(0)
                
                # Adding new jump to price history
                priceHistory.append(priceHistory[-1] + lastJump)
                
            # All other jumps
            else:
                # Getting a new jump
                newJump = self.singleJump(lastJump, method)
                
                # Adding new jump to price history
                priceHistory.append(priceHistory[-1] + lastJump)
                
                # Setting new jump to the last jump
                lastJump = newJump
        
        return priceHistory
    
    
    def plotMonteCarlo(self, df:pd.DataFrame, numPaths:int, plotType:str = "term", compareColumn = "Open") -> None:
        
        try:
            assert(plotType in ["term", "hist"])
        except:
            ValueError("ploType must be either term or hist for terminal or historical distributions")
        
        # Creating a copy of the dataframe
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Naming dictionary
        names = {"hist" : "Historical Price", "term" : "Terminal Price"}

        # All the simulation column
        sim_cols = [c for c in df.columns if c.startswith("Simulation")]
        
        # All historical price values
        all_sim_values = df[sim_cols].to_numpy().ravel()

        # Final values of each simulation (last row)
        terminal_values = df[sim_cols].iloc[-1].values

        # Layouts
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

        ax_paths = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1], sharey=ax_paths)

        # Plotting Simulation
        for col in sim_cols:
            ax_paths.plot(
                df["Date"],
                df[col],
                alpha=0.6,
                linewidth=1,
            )

        # Plotting the Open Price, on top of all the simulations
        ax_paths.plot(
            df["Date"],
            df[compareColumn],
            color="black",
            linewidth=2,
            label="Actual",
            zorder=10
        )

        ax_paths.set_xlabel("Date")
        ax_paths.set_ylabel("Price")
        ax_paths.legend()
        ax_paths.set_title(f"Monte Carlo Simulations with {names[plotType]} Distribution ({numPaths} Simulations)")
        
        # Getting the values to use based off histogram
        if plotType == "term":
            values = terminal_values
        elif plotType == "hist":
            values = all_sim_values

        values = pd.Series(values).dropna()
        
        # Sideways distribution (histogram)
        ax_dist.hist(
            values,
            bins=30,
            orientation="horizontal",
            alpha=0.7,
            density=True
        )
        
        # KDE
        kde = self.__getKDE(values, "SilvermanUni")

        # Getting the x values to fill with y from the KDE
        y_vals = np.linspace(
            values.min(),
            values.max(),
            500
        )

        # Filling the y values from the KDE function using the x values from above
        kde_vals = kde(y_vals)

        # Plotting the kde distribution
        ax_dist.plot(
            kde_vals,
            y_vals,
            linewidth=2,
            label = "Simulated KDE"
        )
        
        if plotType == "term":
            # Expected value of the KDE
            expected_value = np.mean(values)
            
            # Ending value of the actual path
            actual_terminal_value = df[compareColumn].dropna().iloc[-1]
            
            # Plotting the ticks
            ax_dist.plot(0, expected_value, marker='x', markersize=6, clip_on=False, label = "Expected Value")
            ax_dist.plot(0, actual_terminal_value, marker='o', markersize=6, clip_on=False, label = "Actual Value")
        
        elif plotType == "hist":
            # Getting a KDE of the actual data
            actual_KDE = self.__getKDE(df[compareColumn].dropna(), "SilvermanUni")
            
            # Getting the y values of the KDE for a line
            actual_KDE_vals = actual_KDE(y_vals)
            
            # Plotting the actual kde distribution
            ax_dist.plot(
                actual_KDE_vals,
                y_vals,
                linewidth=2,
                label = "Actual KDE"
            )

        # Labeling
        ax_dist.set_xlabel("Frequency")
        ax_dist.tick_params(axis="y", labelleft=False)
        ax_dist.legend()

        # Clean up look
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)

        plt.show()
    
    
    def monteCarloSimulation(self, numPaths: int, plotSimulation:bool, compareColumn = "Open", **kwargs) -> pd.DataFrame:
        
        # Making sure there is a valid column to compare
        assert(compareColumn in ["Open", "High", "Low", "Close"])
        
        # Creating a new dataframe for Monte Carlo paths to be stored
        simulationDataframe = pd.DataFrame(self.tickerData[["Date",compareColumn]].dropna())
        
        # Default functionality
        default = {
            "startingPrice" : simulationDataframe[compareColumn][0], 
            "numSteps" : simulationDataframe.shape[0],
            "plotType" : "term",
            "samplingMethod" : "whole"
        }
        
        # Updating defaults with kwargs
        default.update(kwargs)
        
        simulations = []
        
        # Running over the desired amounts of number of paths to be simulated 
        for j in range(numPaths):
            # Simulating a single path
            singlePath = self.simulatePath(default["startingPrice"], default["numSteps"], method = default["samplingMethod"])
            
            simulations.append(
                pd.Series(singlePath, name=f"Simulation {j}")
                )
        
        # Joining simulation paths together
        simulations_df = pd.concat(simulations, axis=1)
        
        # Adding simulation paths to actual path in the dataframe
        simulationDataframe = pd.concat(
            [simulationDataframe, simulations_df],
            axis=1
        )
        
        # Plotting if called for, else returning the dataframe
        if plotSimulation == True:
            self.plotMonteCarlo(simulationDataframe, numPaths, plotType=default["plotType"], compareColumn=compareColumn)
        
        else:
            return simulationDataframe