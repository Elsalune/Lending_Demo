#!/usr/bin/python

"""Partie1, statistiques descriptives

Functions implémentées :
    bivariate_analysis : afficher le nuage de points de 2 variables
    prepare_data : nettoyage et extraction des variables pour l'analyse stats
    part1 : génération des différents graphiques


"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def bivariate_analysis(data, variable, alpha=0.1, title=''):
    '''Affichage le nuage de points (variable[0], variable[1])

    :param data: DataFrame issue du csv de données
    :param variable: colonne à afficher
    :param alpha: transparence des points à afficher
    :param title: titre du plot
    :return:
    '''
    variableX, variableY = variable[0], variable[1]
    dataviz = pd.concat([data[variableY], data[variableX]], axis=1)
    dataviz.plot.scatter(x=variableX, y=variableY, alpha=alpha, title=title)


def prepare_data(data):
    '''
    Prépare les données pour l'analyse statistique

    Réalise :
        - indexation de la DataFrame selon la date
        - ajout des variables hour, month, year
        - vérifie qu'il n'y a pas de données manquantes

    :param data: DataFrame issue du csv de données
    :return:

    Todo :
        - ajout d'entrées avec les dates manquantes avec interpolation
        - correction des valeurs aberrantes (eg. windspeed).

    '''

    # Index by date
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index("datetime")

    # Empty value checking
    missinfo = (pd.isnull(data)).describe()
    miss_check = (missinfo.loc['freq']==len(data))
    if miss_check.tolist() == [True] * len(data.columns):
        print("No missing data !")
    else:
        print("Missing data !")

    # Create usefull date variables
    data['hour'] = data.index.hour
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['cumulmonth'] = data['month'] + 12*(data['year']-data['year'][0])
    data['weekend'] = (1-data['workingday']) * (1-data['holiday'])
    data['dayName'] = data.index.dayofweek

    # If you add features, don't forget to add them here :
    feature_names = ['hour', 'workingday', 'season', 'holiday', 'weather', 'temp', 'atemp', \
                     'humidity', 'month', 'cumulmonth', 'year', \
                     'weekend', 'dayName']

    return [data, feature_names]


def exploredata(data_path, subpart):
    '''Execute la partie statistiques et affiche les différents graphs.

    :param data_path: chemin vers le csv à analyser.
    :param subpart: sous-partie à analyser parmi :
        > 0 : distribution des variables count, registered et casual.
        > 1 : moyenne de count/registered/casual en fonction des variables indépendantes.
        > 2 : analyse bivariée (matrice de corrélation, nuages de points...)
        > 3 : augmentation globale du trafic moyen et variation saisonnières
        > 4 : analyse horaire
        > 5 : facteurs météorologiques
    :return:
    '''

    data = pd.read_csv(data_path)
    [data, feature_names] = prepare_data(data)

    if subpart==0:
        print("Just a test")
        # Distributions des variables à prédire
        for dep_var in ['registered', 'casual', 'count']:
            print(data[dep_var].describe())
            print("Skewness : %f" % data[dep_var].skew())
            print("Kurtosis %f " % data[dep_var].kurt())
            plt.figure()
            sns.distplot(data[dep_var])

    elif subpart==1:
        # Variable à prédire en fonction de chaque variable indép.
        for X in data.columns:
            if X == 'count' or X =='registered' or X == 'casual':
                continue
            plt.figure()
            mean_byX = data.groupby(X).mean()['count']
            mean_byX.plot(kind='bar', legend=True, color="blue")
            mean_byX = data.groupby(X).mean()['registered']
            mean_byX.plot(kind='bar', legend=True, color="green")
            mean_byX = data.groupby(X).mean()['casual']
            mean_byX.plot(kind='bar', legend=True, color="red")

    elif subpart==2:
        # Analyse bivariée des variables (peu lisible) :
        sns.set()
        vars = ['temp', 'weather', 'humidity', 'windspeed', 'count', 'registered','casual']
        sns.pairplot(data[vars], size = 2.5, plot_kws={'alpha':0.005})
        #plt.show()

        corr_data = data.corr()
        sns.heatmap(corr_data, square=True)

    elif subpart==3:
        # Augmentation globale du trafic, mois par mois :
        count_bymonth = data.groupby('cumulmonth').sum()['count']
        registered_bymonth = data.groupby('cumulmonth').sum()['registered']
        casual_bymonth = data.groupby('cumulmonth').sum()['casual']

        count_bymonth.plot(kind='bar', legend=True, color='red')
        registered_bymonth.plot(kind='bar', legend=True, color='blue')
        casual_bymonth.plot(kind='bar', legend=True, color='green')

        # Même tendance pour les jours travaillés que pour les jours non travaillés :
        data_workingday = data[data.workingday==1]
        reg_workingday = data_workingday.groupby('cumulmonth').mean()['registered']
        data_notworkingday = data[data.workingday==0]
        reg_notworkingday = data_notworkingday.groupby('cumulmonth').mean()['registered']

        reg_workingday.plot(legend=True, label="workingday=1", title="registered")
        reg_notworkingday.plot(legend=True, label="workingday=0", title="registered")

    elif subpart==4:
        # Analyse du facteur hour :
        # Exemple de juin 2012 :
        data_month18 = data[data.cumulmonth==18]
        data_workingday18 = data_month18[data_month18.workingday==1]
        data_noworkingday18 = data_month18[data_month18.workingday==0]
        bivariate_analysis(data_workingday18, ['hour', 'registered'])

        # Sur tous les mois, on normalise par le nombre d'usager dans la journée
        for work in range(0,2):
            tdata = data.copy(deep=True)
            tdata = tdata[tdata.workingday==work]
            day_groups = tdata.groupby(lambda x: x.date())
            tdatanorm = tdata.copy(deep=True)

            tdatanorm['registered'] = day_groups['registered'].transform(lambda x: (x / x.sum()) )
            tdatanorm['casual'] = day_groups['casual'].transform(lambda x: (x / x.sum()) )

            tdata.boxplot(column='registered', by='hour')
            tdatanorm.boxplot(column='registered', by='hour')

            tdata.boxplot(column='casual', by='hour')
            tdatanorm.boxplot(column='casual', by='hour')

    elif subpart==5:
        # Variables météorologiques :
        data.boxplot(column='count', by='weather')
        data.boxplot(column='count', by='temp', rot=90)
        data.boxplot(column='count', by='humidity')
        data.boxplot(column='count', by='windspeed')

    plt.show()