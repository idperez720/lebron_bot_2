#!/usr/bin/env python3

from lebron_bot_2.srv import pickup_service, pickup_serviceResponse
import rospy

def handle_pickup_service(req):
    print("hola")
    print(f'Returning: {req}')
    resp = pickup_serviceResponse()
    resp.answer.state = 1
    resp.answer.pickup_zone = 2
    resp.answer.pickup_zone_x = 4.5
    resp.answer.pickup_zone_y = 3.5
    return resp

def pickup_service_server():
    rospy.init_node('pickup_zone_service_server')
    s = rospy.Service('pickup_service', pickup_service, handle_pickup_service)
    print("Ready to receive")
    rospy.spin()

if __name__ == '__main__':
    pickup_service_server()
