/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

/**
 *
 * @author stuart
 */
public class Datum {

    int iteration;
    double goal;
    double hopfield, jordan, elman, FFNN;
    double chargedPSO, quantumPSO, standardPSO;

    Datum(int epoch) {
        iteration = epoch;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public double getHopfield() {
        return hopfield;
    }

    public void setHopfield(double hopfield) {
        this.hopfield = hopfield;
    }

    public double getJordan() {
        return jordan;
    }

    public void setJordan(double jordan) {
        this.jordan = jordan;
    }

    public double getElman() {
        return elman;
    }

    public void setElman(double elman) {
        this.elman = elman;
    }

    public double getFFNN() {
        return FFNN;
    }

    public void setFFNN(double FFNN) {
        this.FFNN = FFNN;
    }

    public double getChargedPSO() {
        return chargedPSO;
    }

    public void setChargedPSO(double chargedPSO) {
        this.chargedPSO = chargedPSO;
    }

    public double getQuantumPSO() {
        return quantumPSO;
    }

    public void setQuantumPSO(double quantumPSO) {
        this.quantumPSO = quantumPSO;
    }

    public double getGoal() {
        return goal;
    }

    public void setGoal(double goal) {
        this.goal = goal;
    }

    public double getStandardPSO() {
        return standardPSO;
    }

    public void setStandardPSO(double standardPSO) {
        this.standardPSO = standardPSO;
    }

}
