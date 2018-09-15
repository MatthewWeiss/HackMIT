# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:00:01 2018

@author: Matthew and Kyle

The premise of this project is a betting game where participants choose one team for each
week of the NFL season that they predict will win. They can choose each week after seeeing the
results from previous weeks, but they can't reuse a team.

Our code uses ELO data from 538 and their model to calculate each teams probability of winning
each game, and charts the optimal plan through the season to maximize the chances of winning
all 17 weeks. It can be updated after each week with a new ELO file to rechart the remaining plan

The parts that can be changed are ELOFILE and PICKED
"""
#imports
import csv
import os
from pulp import *
import numpy as np
#global variables; number of teams and weeks
TEAMS=32
WEEKS=17
#an array of team abbreviations, such as 'MIN', 'NO', 'NE'
PICKED=[]

#Lists used for the PuLP parts
TEAM_INDICES=[i for i in range(TEAMS)]
WEEK_INDICES=[j for j in range(WEEKS)]

#Files

#Schedule should not be changed in a given season, although it could be changed to backtest if desired
SCHEDFILE='NFL 2018-2019 Schedule.csv'

#File of ELO values, can be changed as weeks pass and elos update
ELOFILE='2018 preseason elo.csv'
#ELOFILE='elo post week one.csv'
def load_data(filename):
    """Reads in the schedule dataset from a file. This should be used to read in a .csv file
    and scrape it"""
    f=open(filename,'r')
    if f.mode=='r':
        data=f.read()
        return data
def fix_data_s(data):
    """This reformats the data into a nice format. If a new data input format is used, this code
    can be reworked without affecting the rest"""
    data=data.split(',')
    for i in range(len(data)):
        item=data[i]
        if item[0]=="\n":
            data[i]=item[1:]
    return data

def build_schedule_dict(data):
    """Builds a dictionary of the schedule from the well formatted data"""
    sched_dict={}
    team_list=[]
    for i in range(TEAMS):
        sched_dict[data[i*(WEEKS+1)]]=data[i*(WEEKS+1)+1:(i+1)*(WEEKS+1)]
        team_list.append(data[i*(WEEKS+1)])
    return sched_dict, team_list
        
def set_up_schedule(filename):
    """Sets up the schedule"""
    return build_schedule_dict(fix_data_s(load_data(filename)))

def build_elo_dict(data):
    """Builds a dictionary mapping team abbreviations to ELO ratings"""
    elo_dict={}
    for i in range(TEAMS):
        elo_dict[data[2*i]]=float(data[2*i+1])
    return elo_dict
        
def set_up_elo(filename):
    """Properly formats the ELO data read in from the csv"""
    return build_elo_dict(fix_data_s(load_data(filename)))

def compute_home_win_prob(home,road, elo_dict, true_home=True):
    """Given two teams, home and road, computes the home team's win probability
    based on 538's model. Can also handle neutral site games"""
    home_elo=elo_dict[home]
    road_elo=elo_dict[road]
    prob=1/(10**(-1*(home_elo-road_elo+65*true_home)/400)+1)
    return prob

def create_wp_dict(sched_dict,elo_dict):
    """Creates a dictionary mapping teams to lists of their win probabilities
    for each game"""
    wp_dict={}
    for team in sched_dict:
        wp_dict[team]=[]
        for game in sched_dict[team]:
            if game=='BYE':
                logprob=-10
            elif game[0]=='@':
                logprob=np.log(1-compute_home_win_prob(game[1:],team,elo_dict))
            elif game[0]=='&':
                logprob=np.log(compute_home_win_prob(team,game[1:],elo_dict,False))
            else:
                logprob=np.log(compute_home_win_prob(team,game,elo_dict))
            wp_dict[team].append(logprob)
    return wp_dict
    
def write_wp_to_file(wp_dict, team_list):
    """Writes the wp_dict to a file so the optimization can be run in other programs
    if desired"""
    os.remove('wp.csv')
    with open('wp.csv','w+') as file:
        writer=csv.writer(file)
        for team in team_list:
            writer.writerow([team]+wp_dict[team])
            
def reformat(wp_dict, team_list):
    """reformates the dictionary of win probabilities into an array"""
    reformatted=[]
    for team in team_list:
        reformatted.append(wp_dict[team])
    return reformatted

def solver(wp_array, team_list, sched_dict, picked=[]):
    """Finds and prints the optimal strategy by solving the MILP"""
    #handles when there are teams that have already been picked
    weeks_gone=len(picked)
    cantuse=[]
    for team in picked:
        for i in range(TEAMS):
            if team_list[i]==team:
                cantuse.append(i)
                break
    #builds model as a maximization
    prob=pulp.LpProblem("Pickem",LpMaximize)
    #our x_ij variables
    xvars=LpVariable.dicts("Picked",[(i,j) for i in TEAM_INDICES for j in WEEK_INDICES],0,1,LpBinary)
    #a dummy variable used to track the objective and print it
    dummy=LpVariable("Solution", None, None, LpContinuous)
    #Objective function maximizes the sums of the logs of the probabilities, thus maximizing the
    #product of the probabilities
    prob+=lpSum([xvars[(i,j)]*wp_array[i][j] for i in TEAM_INDICES for j in WEEK_INDICES])
    
    #Makes sure only one team is picked each week remaining
    for j in range(WEEKS-weeks_gone):      
        prob+=lpSum([xvars[(i,j+weeks_gone)] for i in TEAM_INDICES])==1
    #Makes sure each team is picked at most once
    for i in range(TEAMS):
        prob+=lpSum([xvars[(i,j)] for j in WEEK_INDICES])<=1
    #makes sure we don't pick a team we already picked in a previous week
    for k in cantuse:
        prob+=lpSum([xvars[(k,j)] for j in WEEK_INDICES])==0
    #makes sure we don't make picks for past weeks
    for j in range(weeks_gone):
        prob+=lpSum([xvars[(i,j)] for i in TEAM_INDICES])==0
    #sets the dummy equal to the objective
    prob+=lpSum([xvars[(i,j)]*wp_array[i][j] for i in TEAM_INDICES for j in WEEK_INDICES])==dummy
#    prob+=lpSum([dummy])<=np.log(0.0167)
    
    
    #solves the model
    prob.solve()
    
    #prints the picks for each week, and then the probability of winning after
    for j in WEEK_INDICES:
        for i in TEAM_INDICES:
            if xvars[(i,j)].varValue==1:
                print("Week", j+1, "Pick", team_list[i], "playing", sched_dict[team_list[i]][j])

    print("probability of winning:", np.exp(dummy.varValue))

def main():
#    d=(load_data("2018 preseason elo.csv"))
#    d=d.split(',')
#    print(len(d))
    sched_dict,team_list=set_up_schedule(SCHEDFILE) 
    elo_dict=set_up_elo(ELOFILE)
    wp_dict=create_wp_dict(sched_dict,elo_dict)

#    check=[sum(wp_dict[item]) for item in wp_dict]

    write_wp_to_file(wp_dict,team_list)
    solver(reformat(wp_dict,team_list), team_list, sched_dict, PICKED)
    

    
if __name__=='__main__':
    main()