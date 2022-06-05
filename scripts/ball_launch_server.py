#!/usr/bin/env python3

from lebron_bot_2.srv import ball_service, ball_serviceResponse
import rospy

def handle_ball_service(req):
    print(f'Returning: {req}')
    resp = ball_serviceResponse()
    resp.answer.ball_color = "Red"
    resp.answer.launch_zone = 2
    resp.answer.launch_zone_X = 4.5
    resp.answer.launch_zone_Y = 3.5

    return resp

def ball_service_server():
    rospy.init_node('ball_service_server')
    s = rospy.Service('ball_service', ball_service, handle_ball_service)
    print("Ready to receive")
    rospy.spin()

if __name__ == '__main__':
    ball_service_server()
