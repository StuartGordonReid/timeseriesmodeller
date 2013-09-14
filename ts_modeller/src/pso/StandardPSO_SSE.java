/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pso;

import java.io.File;
import java.util.LinkedList;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.simple.EncogUtility;

/**
 *
 * @author stuart
 */
public class StandardPSO_SSE {

    LinkedList<FFNNParticle> population;
    LinkedList<double[]> vectorPopulation;
    int numInputs, numHidden, numOutputs;
    MLDataSet dataSet;
    double errorGoal = 0.0001;
    double social = 0.75;
    double local = 0.25;
    CalculateScore score;
    public FFNNParticle gbest;

    public StandardPSO_SSE(int populationSize, int in, int hidden, int out, MLDataSet dataS) {
        numInputs = in;
        numHidden = hidden;
        numOutputs = out;

        dataSet = dataS;
        score = new TrainingSetScore(dataSet);

        population = new LinkedList();
        for (int i = 0; i < populationSize; i++) {
            FFNNParticle particle = new FFNNParticle(numInputs, numHidden, numOutputs);
            population.add(particle);
        }
    }

    public void test() {
        for (int i = 0; i < 5000; i++) {
            iteration();
            //System.out.println(gbest.getParticle().dumpWeights());
            System.out.println(getScore(gbest));
        }
    }

    public void iteration() {
        setGlobalBestParticle();
        calculateVectors();
        updateLocalBests();
        setVelocities();
        setPositions();
        updatePopulation();
    }

    private void calculateVectors() {
        vectorPopulation = new LinkedList();
        for (FFNNParticle particle : population) {
            int numConnections = particle.getParticle().encodedArrayLength();
            double[] weights = new double[numConnections];
            particle.getParticle().encodeToArray(weights);
            vectorPopulation.add(weights);
        }
    }

    private void updatePopulation() {
        for (int i = 0; i < population.size(); i++) {
            population.get(i).getParticle().decodeFromArray(vectorPopulation.get(i));
        }
    }

    private void setPositions() {
        double[] gbest_vector = getGlobalBest();

        for (int i = 0; i < population.size(); i++) {

            double velocity = population.get(i).velocity;
            double[] particle = vectorPopulation.get(i);

            for (int j = 0; j < particle.length; j++) {
                if (gbest_vector[j] > particle[j]) {
                    particle[j] -= velocity;
                } else {
                    particle[j] += velocity;
                }
            }

            vectorPopulation.set(i, particle);
        }
    }

    private void setVelocities() {
        double gbest_score = getScore(gbest);

        for (FFNNParticle particle : population) {
            double lbest_score = particle.localBestScore;
            double particle_score = getScore(particle);

            double s = social * (gbest_score - particle_score);
            double c = local * (lbest_score - particle_score);
            double v = particle.velocity + s + c;

            particle.velocity = v * 0.5;
        }
    }

    private void setGlobalBestParticle() {
        gbest = null;
        double gbest_score = Double.MAX_VALUE;

        for (FFNNParticle particle : population) {

            double particle_score = getScore(particle);
            if (particle_score < gbest_score) {
                gbest = particle;
                gbest_score = particle_score;
            }
        }
    }

    private double[] getGlobalBest() {
        int numConnections = gbest.getParticle().encodedArrayLength();
        double[] weights = new double[numConnections];
        gbest.getParticle().encodeToArray(weights);
        return weights;
    }

    private void updateLocalBests() {
        for (FFNNParticle particle : population) {
            double particle_score = getScore(particle);

            if (particle_score < particle.getLocalBestScore()) {
                int numConnections = particle.getParticle().encodedArrayLength();
                double[] weights = new double[numConnections];

                particle.getParticle().encodeToArray(weights);
                particle.localBest = weights;
                particle.setLocalBestScore(particle_score);
            }
        }
    }

    public double getScore(FFNNParticle particle) {
        MLTrain trainAlt = new NeuralSimulatedAnnealing(particle.getParticle(), score, 10, 2, 100);
        MLTrain trainMain = new Backpropagation(particle.getParticle(), dataSet, errorGoal, 0.0);

        trainMain.addStrategy(new Greedy());
        trainMain.addStrategy(new HybridStrategy(trainAlt));

        for (int i = 0; i < 3; i++) {
            trainMain.iteration();
        }

        //System.out.println(trainMain.getError());
        return trainMain.getError();
    }

    public static void main(String args[]) {
        String name = "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/BRAZIL_train.egb";
        MLDataSet data = EncogUtility.loadEGB2Memory(new File(name));
        StandardPSO_SSE pso = new StandardPSO_SSE(30, 22, 30, 1, data);
        pso.test();
    }
}
