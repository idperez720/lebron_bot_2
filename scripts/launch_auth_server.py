#!/usr/bin/env python3

from lebron_bot_2.srv import launch_auth_service, launch_auth_serviceResponse
import rospy

def handle_launch_auth_service(req):
    #print(f'Returning: {req}')
    print(req.message_decode)
    resp = launch_auth_serviceResponse()
    if req.message_decode == "Hoy":
        resp.auth = 1
    else:
        resp.auth = 0

    return resp

def launch_auth_service_server():
    rospy.init_node('launch_auth_service_server')
    s = rospy.Service('launch_auth_service', launch_auth_service, handle_launch_auth_service)
    print("Ready to receive")
    rospy.spin()

if __name__ == '__main__':
    launch_auth_service_server()