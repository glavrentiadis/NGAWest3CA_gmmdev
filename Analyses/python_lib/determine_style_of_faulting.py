#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 08:29:38 2023

@author: glavrent
"""

def BA07sof(p_axis_plunge, t_axis_plunge):
    '''Boore and Atkinson 2007'''
    
    if   p_axis_plunge <= 40 and t_axis_plunge <= 40: sof = 'strike-slip'
    elif p_axis_plunge <= 40 and t_axis_plunge >  40: sof = 'reverse'
    elif p_axis_plunge >  40 and t_axis_plunge <= 40: sof = 'normal'
    else: sof = 'unclassified'

    return sof

def Z92sof(p_axis_plunge, t_axis_plunge):
    '''Zoback 1992'''

    if   p_axis_plunge <  40 and t_axis_plunge <= 20: sof = 'strike-slip'
    elif p_axis_plunge <= 20 and t_axis_plunge <  40: sof = 'strike-slip'
    elif p_axis_plunge <= 35 and t_axis_plunge >= 52: sof = 'reverse'
    elif p_axis_plunge >= 52 and t_axis_plunge <= 35: sof = 'normal'
    elif p_axis_plunge <= 20 and t_axis_plunge >  40 and t_axis_plunge <  52: sof = 'reverse-oblique'
    elif p_axis_plunge >  40 and p_axis_plunge <  52 and t_axis_plunge <= 20: sof = 'normal-oblique'
    else: sof = 'unclassified'
    
    return sof

def W94sof(rake_angle):
    '''Wells and Coppersmith 1994'''

    if   abs(rake_angle) < 45 or abs(rake_angle) > 135: sof = 'strike-slip'
    elif rake_angle <=  135 and rake_angle >=  45: sof = 'reverse'
    elif rake_angle >= -135 and rake_angle <= -45: sof = 'normal'
    else: sof  = 'unclassified'

    return sof

