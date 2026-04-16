import xml.etree.ElementTree as ET

urdf_path = "p73_walker.urdf"

tree = ET.parse(urdf_path)
root = tree.getroot()

total_mass = 0.0

for link in root.findall("link"):
    inertial = link.find("inertial")
    if inertial is not None:
        mass = inertial.find("mass")
        if mass is not None:
            m = float(mass.attrib["value"])
            print(f"{link.attrib['name']}: {m} kg")
            total_mass += m

print(f"\nTotal mass: {total_mass} kg")