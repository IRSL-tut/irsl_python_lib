import irsl.sampling.so3 as so3
exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())

def sampleS2Point(level=0, radius=1.0):
    """return list of points
    """
    res=so3.sampleS2(level)
    ret = []
    for theta, pi in zip(res[0] - math.pi/2, res[1]):
        ## print(theta, pi)
        cds = coordinates()
        cds.rotate(pi, coordinates.Z)
        cds.rotate(theta, coordinates.Y)
        ret.append( cds.x_axis * radius )
    return ret

def sampleSO3Coords(level=0, radius=1.0):
    """return list of coordinates
    """
    res=so3.sampleSO3(level)
    lst = [ coordinates(r) for r in res ]
    for l in lst:
        l.translate(fv(0, 0, radius))
    return lst

#
# import irsl_cnoid.sampling.so3 as so3
# di = DrawInterface()
# sp=mkshapes.makeSphere(1.0, transparent=0.2, color=[0, 1, 1], DivisionNumber=48)
# di.addObject(sp)
#### Equally space sampling (on sphere)
# res=so3.sampleS2Point(level=1)
# pts=mkshapes.makePoints(res)
# di.addObject(pts)
#### Uniformly sampling 3D rotation
# res=so3.sampleSO3Coords(level=1)
# cds=[ mkshapes.makeCoords(coords=r, length=0.3) for r in res ]
# di.addObjects(cds)
#
