/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

import java.io.File;
import java.io.FileWriter;
import java.util.LinkedList;

/**
 *
 * @author stuart
 */
public class Reporter {

    String dataFile;
    LinkedList<Datum> data;
    int epochs;

    Reporter(int iter, String file) {
        epochs = iter;
        data = new LinkedList();
        for (int i = 0; i < epochs; i++) {
            data.add(new Datum(i));
        }
        dataFile = file;
    }

    public void add(Datum dat) {
        data.add(dat);
    }

    public Datum get(int epoch) {
        return data.get(epoch);
    }

    public void printToCSV(String fileName) {
        try {
            String fullName = fileName;
            File f = new File(fullName);
            FileWriter fwriter = new FileWriter(fullName);

            String lineOut = "Iteration,Elman,Hopfield,Jordan,FFNN,QuantumPSO,ChargedPSO,StandardPSO,Goal";
            fwriter.write(lineOut);

            for (Datum d : data) {
                lineOut = "" + d.getIteration() + ",";
                lineOut += d.getElman() + ",";
                lineOut += d.getHopfield() + ",";
                lineOut += d.getJordan() + ",";
                lineOut += d.getFFNN() + ",";
                lineOut += d.getQuantumPSO() + ",";
                lineOut += d.getChargedPSO() + ",";
                lineOut += d.getStandardPSO() + ",";
                lineOut += d.getGoal() + "\n";
                fwriter.write(lineOut);
                fwriter.flush();
            }
        } catch (Exception err) {
            err.printStackTrace();
        }
    }

}
