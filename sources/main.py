#!/usr/bin/python

"""

Usage:
    main.py [-h] -p {1,2} [-sp {0,1,2,3,4}] [-f X]
                            [-ml {train,crossvalidate,trainandtest,featureimportance}]
                            [-train TRAIN] [-test TEST] [-o O]

Optional arguments:
    -h, --help            show this help message and exit
    -p {1,2}              Partie à lancer : 1/ Stats. 2/ Machine learning
    -sp {0,1,2,3,4}       Sous-partie à lancer pour la partie stats. Evite
                        l'affiche de nombreux graphiques.
    -f X                  Afficher uniquement la figure n°X du rapport
    -ml {train,crossvalidate,trainandtest,featureimportance}
                        Procédure de ML à lancer.
    -train TRAIN          Chemin vers les données csv d'entrainement.
    -test TEST            Chemin vers les données csv de tests.
    -o O                  Chemin vers le csv de résutlats.


"""
import machine_learning as ml
import data_exploration as dataexp
import argparse

parser = argparse.ArgumentParser(description='Kaggle Bike sharing demand')

parser.add_argument('-p', type=int, nargs='?', choices=(1,2), required=True, \
                    help='Partie à lancer  : 1/ Stats. 2/ Machine learning' )

parser.add_argument('-sp', type=int, nargs='?', choices=(0,1,2,3,4,5), default=0, \
                    help='''Sous-partie à lancer pour la partie stats parmi :
                        0=Distribution des variables count, registered et casual.
                        1=Moyenne de count/registered/casual en fonction des variables indépendantes.
                        2=Analyse bivariée (matrice de corrélation, nuages de points...)
                        3=Augmentation globale du trafic moyen et variation saisonnières.
                        4=Analyse horaire.
                        5=Facteurs météorologiques.''' )


parser.add_argument('-train', type=str, nargs='?', default='data/data.csv', \
                    help='Chemin vers les données csv de tests.' )

parser.add_argument('-test', type=str, nargs='?', default='data/test.csv', \
                    help='Chemin vers les données csv de tests.' )

parser.add_argument('-o', type=str, nargs='?', default='data/submissions.csv', \
                    help='''Chemin vers le csv de résutlats.''' )

parser.add_argument('-cv', type=str, nargs='?', default='cv_timeseries', \
                    choices=['cv_random', 'cv_timeseries'], \
                    help='''Type de sampling pour la crossvalidation.''' )

parser.add_argument('-proc', type=str, nargs='?', default='crossvalidate', \
                    choices=['crossvalidate', 'train', 'test', 'trainandtest', 'featureimportance'], \
                    help='''Type de procedure de machine learning.''' )

parser.add_argument('-e', type=str, nargs='?', default='my_estimator', \
                    choices=['random_forests', 'part2_estimator',], \
                    help='''Modèle de prédiction à lancer.''' )

parser.add_argument('-v', type=int, nargs='?', choices=(0,1), required=False, \
                    default=0, help='Verbose.' )

args = parser.parse_args()
print(args)


if args.p == 1:
    # Statistiques descriptives :
    dataexp.exploredata(args.train, args.sp)
elif args.p == 2:
    # Machine learning :
    ml.predict(args.proc, args.train, test_path=args.test, result_path=args.o, \
                cv_sampling=args.cv, estimator=args.e, verbose=args.v)