/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

import java.io.File;
import java.text.DecimalFormat;

import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.simple.EncogUtility;

import pso.ChargedPSO;
import pso.FFNNParticle;
import pso.QuantumPSO;
import pso.StandardPSO;
import recurrent.Elman;
import recurrent.Hopfield;
import recurrent.Jordan;

/**
 *
 * @author stuart
 */
public class Simulator {

    String[] datasets;
    Reporter[] results;
    double errorGoal = 0.0001;
    boolean stuart = true;
    DecimalFormat df = new DecimalFormat("#0.0000#");

    Simulator(String[] dataFiles, int iterations) {
        datasets = dataFiles;
        results = new Reporter[dataFiles.length];
        for (int i = 0; i < dataFiles.length; i++) {
            results[i] = new Reporter(iterations, dataFiles[i]);
        }
    }

    public void start() {
        try {
            System.out.println("Running standard FFNN");
            simulateFFNN();
            System.out.println("Running standard Elman RNN");
            simulateElman();
            System.out.println("Running standard Jordan RNN");
            simulateJordan();
        } catch (Exception experimentException) {
            experimentException.printStackTrace();
        }

        for (int i = 0; i < results.length; i++) {
            if (stuart) {
                System.out.println("Output configured to stuart");
                switch (i) {
                    case 0:
                        results[i].printToCSV("Ouput_Brazil.csv");
                        break;
                    case 1:
                        results[i].printToCSV("Ouput_Russia.csv");
                        break;
                    case 2:
                        results[i].printToCSV("Ouput_India.csv");
                        break;
                    case 3:
                        results[i].printToCSV("Ouput_China.csv");
                        break;
                    case 4:
                        results[i].printToCSV("Ouput_SA.csv");
                        break;
                }
            } else {
                System.out.println("Output configured to simon");
            }
        }
    }

    public void simulateElman() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            Elman nn = new Elman(22, 30, 1);
            CalculateScore score = new TrainingSetScore(dataSet);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), dataSet, errorGoal, 0.0);
            StopTrainingStrategy stop = new StopTrainingStrategy();

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));
            trainMain.addStrategy(stop);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = trainMain.getError();;

                //Save the value in results
                results[i].get(j).setElman(value);
                trainMain.iteration();
            }
        }
    }

    public void simulateHopfield() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            Hopfield nn = new Hopfield();
            CalculateScore score = new TrainingSetScore(dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = 0.0;

                //Save the value in results
                results[i].get(j).setHopfield(value);
            }
        }
    }

    public void simulateJordan() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            Jordan nn = new Jordan(22, 30, 1);
            CalculateScore score = new TrainingSetScore(dataSet);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), dataSet, errorGoal, 0.0);
            StopTrainingStrategy stop = new StopTrainingStrategy();

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));
            trainMain.addStrategy(stop);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = trainMain.getError();

                //Save the value in results
                results[i].get(j).setJordan(value);
                trainMain.iteration();
            }
        }
    }

    public void simulateFFNN() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            FFNNParticle nn = new FFNNParticle(22, 30, 1);
            CalculateScore score = new TrainingSetScore(dataSet);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), dataSet, errorGoal, 0.0);
            StopTrainingStrategy stop = new StopTrainingStrategy();

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));
            trainMain.addStrategy(stop);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = trainMain.getError();

                //Save the value in results
                results[i].get(j).setFFNN(value);
                trainMain.iteration();
            }
        }
    }

    public void simulateChargedPSO() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            ChargedPSO nn = new ChargedPSO();
            CalculateScore score = new TrainingSetScore(dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = 0.0;

                //Save the value in results
                results[i].get(j).setChargedPSO(value);
            }
        }
    }

    public void simulateQuantumPSO() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            QuantumPSO nn = new QuantumPSO();
            CalculateScore score = new TrainingSetScore(dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; i++) {

                double value = 0.0;

                //Save the value in results
                results[i].get(j).setQuantumPSO(value);
            }
        }
    }

    public void simulateStandardPSO() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            StandardPSO nn = new StandardPSO();
            CalculateScore score = new TrainingSetScore(dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = 0.0;

                //Save the value in results
                results[i].get(j).setStandardPSO(value);
            }
        }
    }
}
