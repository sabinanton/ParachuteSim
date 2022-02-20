import numpy as np


def createPorosityProperties(Normals, D, F):
    e1x = Normals[0]
    e1y = Normals[1]
    e1z = Normals[2]
    e2x = Normals[3]
    e2y = Normals[4]
    e2z = Normals[5]

    f = open('porosityProperties', 'w+')
    f.write("FoamFile\n")
    f.write("{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n")
    f.write("\tlocation\t\"constant\";\n\tobject\tporosityProperties;\n}\n\n")

    for i in range(len(e1x)-1):

        f.write("porosity" + str(i) + '\n')
        f.write("{\n\ttype\t\tDarcyForchheimer;\n\n\tcellZone\tpanel" + str(i) + ";\n")
        f.write("\n\td\t(" + str(D[0]) + " " + str(D[1]) + " " + str(D[2]) + ");\n")
        f.write("\tf\t(" + str(F[0]) + " " + str(F[1]) + " " + str(F[2]) + ");\n")
        f.write("\n\tcoordinateSystem\n\t{\n\t\torigin\t(0 0 0);\n\t\te1\t(" + str(e1x[i]) + " " + str(e1y[i]) + " " + str(e1z[i]) + ");\n")
        f.write("\t\te2\t(" + str(e2x[i]) + " " + str(e2y[i]) + " " + str(e2z[i]) + ");\n\t}\n}\n")

def createTopoSet(numFiles):
    f = open('topoSetDict', 'w+')

    f.write("FoamFile\n")
    f.write("{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n")
    f.write("\tlocation\t\"constant\";\n\tobject\ttopoSetDict;\n}\n\nactions\t(\n")

    for i in range(numFiles-1):

        f.write("\t{\n\t\tname\t\tpanel" + str(i) + ";\n\t\ttype\tcellZoneSet;\n")
        f.write("\t\taction\tnew;\n\t\tsource\tsurfaceToCell;\n\t\tsourceInfo\n\t\t\t{\n\t\t\t\tfile\t\"constant/triSurface/STL_Files/panel" + str(i) + ".stl\";\n")
        f.write("\t\t\t\tuseSurfaceOrientation\tfalse;\n\t\t\t\toutsidePoints\t();\n\t\t\t\tincludeInside\tfalse;\n\t\t\t\tincludeOutside\tfalse;\n\t\t\t\tincludeCut\ttrue;\n\t\t\t\tinterpolate\ttrue;\n")
        f.write("\t\t\t\tnearDistance\t0.005;\n\t\t\t\tcurvature\t-1;\n\t\t\t}\n\t}\n")
    f.write(");\n\n")

def createFvOptions(numFiles):
    f = open('fvOptions', 'w+')

    f.write("FoamFile\n")
    f.write("{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n")
    f.write("\tlocation\t\"constant\";\n\tobject\tfvOptions;\n}\n\n")

    for i in range(numFiles-1):
        f.write("source" + str(i) + "\n{\n\ttype\tfixedTemperatureConstraint;\n\n\tselectionMode\tcellZone;\n\tcellZone\tpanel" + str(i) + ";\n")
        f.write("\n\tmode\tuniform;\n\ttemperature\t288;\n}\n")

