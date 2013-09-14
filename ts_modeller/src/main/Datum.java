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
    double jordanSSE, elmanSSE, FFNNSSE;
    double chargedPSOSSE, quantumPSOSSE, standardPSOSSE;
    double jordanPE, elmanPE, FFNNPE;
    double chargedPSOPE, quantumPSOPE, standardPSOPE;

    Datum(int epoch) {
        iteration = epoch;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public double getGoal() {
        return goal;
    }

    public void setGoal(double goal) {
        this.goal = goal;
    }

    public double getJordanSSE() {
        return jordanSSE;
    }

    public void setJordanSSE(double jordanSSE) {
        this.jordanSSE = jordanSSE;
    }

    public double getElmanSSE() {
        return elmanSSE;
    }

    public void setElmanSSE(double elmanSSE) {
        this.elmanSSE = elmanSSE;
    }

    public double getFFNNSSE() {
        return FFNNSSE;
    }

    public void setFFNNSSE(double FFNNSSE) {
        this.FFNNSSE = FFNNSSE;
    }

    public double getChargedPSOSSE() {
        return chargedPSOSSE;
    }

    public void setChargedPSOSSE(double chargedPSOSSE) {
        this.chargedPSOSSE = chargedPSOSSE;
    }

    public double getQuantumPSOSSE() {
        return quantumPSOSSE;
    }

    public void setQuantumPSOSSE(double quantumPSOSSE) {
        this.quantumPSOSSE = quantumPSOSSE;
    }

    public double getStandardPSOSSE() {
        return standardPSOSSE;
    }

    public void setStandardPSOSSE(double standardPSOSSE) {
        this.standardPSOSSE = standardPSOSSE;
    }

    public double getJordanPE() {
        return jordanPE;
    }

    public void setJordanPE(double jordanPE) {
        this.jordanPE = jordanPE;
    }

    public double getElmanPE() {
        return elmanPE;
    }

    public void setElmanPE(double elmanPE) {
        this.elmanPE = elmanPE;
    }

    public double getFFNNPE() {
        return FFNNPE;
    }

    public void setFFNNPE(double FFNNPE) {
        this.FFNNPE = FFNNPE;
    }

    public double getChargedPSOPE() {
        return chargedPSOPE;
    }

    public void setChargedPSOPE(double chargedPSOPE) {
        this.chargedPSOPE = chargedPSOPE;
    }

    public double getQuantumPSOPE() {
        return quantumPSOPE;
    }

    public void setQuantumPSOPE(double quantumPSOPE) {
        this.quantumPSOPE = quantumPSOPE;
    }

    public double getStandardPSOPE() {
        return standardPSOPE;
    }

    public void setStandardPSOPE(double standardPSOPE) {
        this.standardPSOPE = standardPSOPE;
    }

}
