/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

import java.util.LinkedList;

/**
 *
 * @author stuart
 */
public class Datum {

    int iteration;
    double goal;

    LinkedList<Double> jordanSSE, jordanPE;
    LinkedList<Double> elmanSSE, elmanPE;
    LinkedList<Double> FFNNSSE, FFNNPE;
    LinkedList<Double> standardPSOSSE, standardPSOPE;
    LinkedList<Double> chargedPSOSSE, chargedPSOPE;

    Datum(int epoch) {
        iteration = epoch;
        jordanSSE = new LinkedList();
        jordanPE = new LinkedList();
        elmanSSE = new LinkedList();
        elmanPE = new LinkedList();
        FFNNSSE = new LinkedList();
        FFNNPE = new LinkedList();
        standardPSOSSE = new LinkedList();
        standardPSOPE = new LinkedList();
        chargedPSOSSE = new LinkedList();
        chargedPSOPE = new LinkedList();
    }

    public double average(LinkedList<Double> list) {
        double total = 0.0;
        for (Double d : list) {
            total += d;
        }
        return total / list.size();
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public double getAvgJordanSSE() {
        return average(jordanSSE);
    }

    public double getAvgJordanPE() {
        return average(jordanPE);
    }

    public double getAvgElmanSSE() {
        return average(elmanSSE);
    }

    public double getAvgElmanPE() {
        return average(elmanPE);
    }

    public double getAvgFFNNSSE() {
        return average(FFNNSSE);
    }

    public double getAvgFFNNPE() {
        return average(FFNNPE);
    }

    public double getAvgStandardPSOSSE() {
        return average(standardPSOSSE);
    }

    public double getAvgStandardPSOPE() {
        return average(standardPSOPE);
    }

    public double getAvgChargedPSOSSE() {
        return average(chargedPSOSSE);
    }

    public double getAvgChargedPSOPE() {
        return average(chargedPSOPE);
    }
}
