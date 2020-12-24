#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
}

- (IBAction)onBtnTouchedInside:(id)sender {
    self.textLabel.text = @"button pressed.";
}


@end
