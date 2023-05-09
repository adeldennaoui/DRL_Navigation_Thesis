#!/usr/bin/env python3

class anti_collision(object):
    def __init__(self, x_m, y_m, x_f, y_f):
        self.x_m = x_m
        self.y_m = y_m
        self.x_f = x_f
        self.y_f = y_f
    def avoiding_collisions(self):
        if self.x_m > self.x_f:
            goal_x = self.x_m + 1.0
            if self.y_m > self.y_f:
                goal_y = self.y_m + 1.0
            else:
                goal_y = self.y_m - 1.0
        else:
            goal_x = self.x_m - 1.0
            if self.y_m > self.y_f:
                goal_y = self.y_m + 1.0
            else:
                goal_y = self.y_m - 1.0
        return goal_x, goal_y

# PAY ATTENTION!
# Draft solution for anti-collision system in the case of 3 robots that are close to each other!
class anti_collision_3way(object):
    def __init__(self, x_m, y_m, x_f1, y_f1, x_f2, y_f2):
        self.x_m = x_m
        self.y_m = y_m
        self.x_f1 = x_f1
        self.y_f1 = y_f1
        self.x_f2 = x_f2
        self.y_f2 = y_f2
    def avoiding_collisions(self):
        if self.x_m > self.x_f1:
            if self.x_m > self.x_f2:
                goal_x = self.x_m + 1.0
                if self.y_m > self.y_f1:
                    if self.y_m > self.y_f2:
                        goal_y = self.y_m
                    else:
                        goal_y = self.y_m
                else:
                    if self.y_m > self.y_f2:
                        goal_y = self.y_m
                    else:
                        goal_y = self.y_m
            else:
                goal_x = self.x_m
                if self.y_m > self.y_f1:
                    if self.y_m > self.y_f2:
                        goal_y = self.y_m + 1.0
                    else:
                        if self.y_f1 > self.y_f2:
                            goal_y = self.y_f1 + 1.0
                        else:
                            goal_y = self.y_f2 + 1.0
                else:
                    if self.y_m > self.y_f2:
                        if self.y_f1 > self.y_f2:
                            goal_y = self.y_f1 + 1.0
                        else:
                            goal_y = self.y_f2 + 1.0
                    else:
                        goal_y = self.y_m - 1.0
        else:
            if self.x_m > self.x_f2:
                goal_x = self.x_m
                if self.y_m > self.y_f1:
                    if self.y_m > self.y_f2:
                        goal_y = self.y_m + 1.0
                    else:
                        if self.y_f1 > self.y_f2:
                            goal_y = self.y_f1 + 1.0
                        else:
                            goal_y = self.y_f2 + 1.0
                else:
                    if self.y_m > self.y_f2:
                        if self.y_f1 > self.y_f2:
                            goal_y = self.y_f1 + 1.0
                        else:
                            goal_y = self.y_f2 + 1.0
                    else:
                        goal_y = self.y_m - 1.0
            else:
                goal_x = self.x_m - 1.0
                if self.y_m > self.y_f1:
                    if self.y_m > self.y_f2:
                        goal_y = self.y_m + 1.0
                    else:
                        goal_y = self.y_m
                else:
                    if self.y_m > self.y_f2:
                        goal_y = self.y_m
                    else:
                        goal_y = self.y_m - 1.0
        return goal_x, goal_y