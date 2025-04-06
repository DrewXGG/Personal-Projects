import asyncio
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.sensor import Sensor
from viam.components.camera import Camera
from viam.components.board import Board
from viam.components.motor import Motor
from viam.components.base import Base
from viam.components.encoder import Encoder
from viam.components.movement_sensor import MovementSensor
from viam.services.vision import VisionClient
from viam.services.slam import SLAMClient
from viam.media.utils.pil import pil_to_viam_image, viam_to_pil_image

base = Base.from_robot(machine, "viam_base")
camera_name = "cam"


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='7h3hki656kbdmra4idxgb7o9qh88tzcd',
        api_key_id='cf441be0-b1ac-4132-a91e-b521f390502e'
    )
    return await RobotClient.at_address("rover15-main.ulfzthvv99.viam.cloud", opts)

def leftOrRight(detections, midpoint):
    largest_area = 0
    largest = {"x_max": 0, "x_min": 0, "y_max": 0, "y_min": 0}
    if not detections:
        print("nothing detected :(")
        return -1
    for d in detections:
        a = (d.x_max - d.x_min) * (d.y_max-d.y_min)
        if a > largest_area:
            a = largest_area
            largest = d
    centerX = largest.x_min + largest.x_max/2
    if centerX < midpoint-midpoint/6:
        return 0  # on the left
    if centerX > midpoint+midpoint/6:
        return 2  # on the right
    else:
        return 1  # basically centered
    
async def search_for_object(base, search_spin_speed):
    print("Searching for object")
    await base.spin(search_spin_speed, 300)
    

async def tracking():
    machine = await connect()
    spinNum = 5         # when turning, spin the motor this much
    straightNum = 300    # when going straight, spin motor this much
    numCycles = 200      # run the loop X times
    vel = 500            # go this fast when moving motor
    more = 15           # turn more (a larger angle) if not detected
    camera_name = "cam"
    camera = Camera.from_robot(machine, "cam")
    frame = await camera.get_image(mime_type="image/jpeg")

    # Convert to PIL Image
    pil_frame = viam_to_pil_image(frame)

    # Grab the vision service for the detector huan cheng objectDetector
    my_detector = VisionClient.from_robot(machine, "vision-15")
    
    for i in range(numCycles):
        detections = await my_detector.get_detections_from_camera(camera_name)

        answer = leftOrRight(detections, pil_frame.size[0]/2)
        if answer == 0:
            print("left")
            await base.spin(spinNum, vel)     # CCW is positive
            await base.move_straight(straightNum, vel)
        if answer == 1:
            print("center")
            await base.move_straight(straightNum, vel)
        if answer == 2:
            print("right")
            await base.spin(-spinNum, vel)
        else:
            print("No object detected, starting searching")
            # While the object is not detected (-1), start searching
            while answer == -1:
                # Start with a smaller spin, then gradually increase the search area
                await base.spin(more, vel)
                
                detections = await my_detector.get_detections_from_camera(camera_name)

                answer = leftOrRight(detections, pil_frame.size[0] / 2)

        # If object is detected, resume normal tracking
        if answer != -1:
            print("Object detected. Resuming tracking.")
    await machine.close()        

async def person_detect(detector, base):
    while (True):
        # look for a bottle
        found = False
        #global base_state
        print("will detect")
        detections = await detector.get_detections_from_camera(camera_name)
        for d in detections:
            if d.confidence > .7:
                print(d.class_name)
                # specify it is just the person we want to detect
                #如果不是要找的 处理障碍 
                if (d.class_name == "Person"):
                    found = True
        if (found):
            print("I see a person")
            # first manually call gather_obstacle_readings -
            # don't even start moving if someone in the way
            #distances = await gather_obstacle_readings(sensors)
            #if all(distance > 0.4 for distance in distances):
                #print("will move straight")
                #base_state = "straight"
                #await base.move_straight(distance=800, velocity=250)
                #base_state = "stopped"
            tracking()        
        else:
            print("I will turn and look for a bottle")
            #base_state = "spinning"
            await base.spin(45, 45)
            await base.move_straight(distance=100, velocity=200)
            await base.spin(-45, 45)
            #base_state = "stopped"

        await asyncio.sleep(2)

async def main():
    robot = await connect()
    rover_base = Base.from_robot(robot, 'viam_base')
    detector = VisionClient.from_robot(robot, name=detector_name)
    await person_detect()
    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())