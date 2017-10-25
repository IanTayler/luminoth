from luminoth.models.retina.subnet import Subnet


class ClassSubnet(Subnet):
    def __init__(self, config, num_anchors, num_classes, name='box_subnet'):
        num_final_chns = num_anchors * num_classes
        super(ClassSubnet, self).__init__(config, num_final_chns, name=name)
        self._config = config
