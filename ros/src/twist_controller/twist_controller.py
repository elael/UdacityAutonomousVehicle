import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle) # 0.1 m/s is the lowest speed of the car
        
        # PID gain parameters
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # Minimum throttle value
        mx = 0.2 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        #the velocity data that's coming from the messages is noisy, so low-pass filter is used to filter out all the high frequency noise in the velocity data
        tau = 0.5 # 1/(2*pi*tau) = cut-off frequency
        ts = .02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()
   
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        # If the vehicle is just sitting and in our pid we have an integral term and the DBW is enabled, then the intergral
        # term will be just accumulating error
        # So, if drive-by-wire is not enabled, then reset the controller
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        # Filter the current velocity from /current_velocity topic using the low-pass filter
        current_vel = self.vel_lpf.filt(current_vel)
        
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        # If car is supposed to remain stopped
        # Carla has an automatic transmission, which means the car will roll forward if no brake and no throttle is applied
        # So, a minimum brake force to prevent Carla from rolling has to be applied
        if linear_vel == 0 and current_vel<0.1:
            throttle = 0
            brake = 700 # N-m - to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2
            
        # If we need to decelerate
        elif throttle < 0.1 and vel_error <0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Braking torque in N-m
            
        return throttle, brake, steering
            
        #return 1., 0., 0.
