#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace cv;
using namespace cv::xfeatures2d;

struct SURFDetector
{
    Ptr<Feature2D> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};


int count_good_matches(std::vector<DMatch>& matches, double min_distance_threshold)
{
    int good_matches_count = -1;

    // Sort matches
    std::sort(matches.begin(), matches.end());

    //Good matches vector
    std::vector< DMatch > good_matches;

    // Add good matches upto threshold value
    for(int i = 0; i < matches.size(); i++)
    {
        if(matches[i].distance > min_distance_threshold)
            break;
        good_matches.push_back( matches[i] );
    }

    good_matches_count = good_matches.size();

    return good_matches_count;
}

int run_flash_summ(int argc, char* argv[])
{
    //handle input
    const char* keys =
            "{ h help        |            | print help message    }"
            "{ i input       | video.mpg  | specify input video   }"
            "{ o output      | ./output/  | output folder path   }"
            "{ s sensitivity | 0.4        | sensitivity threshold }"
            "{ n noise       | 0.8        | noise threshold }"
            "{ d distance    | 0.2        | matches distance threshold }"
            "{ t interval    | 30         | interval for average matches count }"
            "{ e step        | 3          | distance between two processed frames }"
            "{ v vervose     |            | print internal values }"
            "{ g gui         |            | display video and GUI interface }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        std::cout << "Usage: lfovs [options]" << std::endl;
        std::cout << "Available options:" << std::endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }
    bool verbose=0;
    bool display_video=0;

    //handle user preferences
    if(cmd.get<std::string>("v") == "true")
        verbose=true;
    else
        verbose=false;

    if(cmd.get<std::string>("g") == "true")
        display_video=true;
    else
        display_video=false;

    //handle input
    std::string input_video_path = cmd.get<std::string>("i");
    if(verbose) std::cout << "input video:" << input_video_path << std::endl;

    VideoCapture capture(input_video_path); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened())
        capture.open(0); //open camera 0
    if (!capture.isOpened()) {
        std::cerr << "Failed to open the video device, video file or image sequence!\n" << std::endl;
        return EXIT_FAILURE;
    }

    //handle output path and file
    std::string output_path = cmd.get<std::string>("o");
    if(verbose) std::cout << output_path << std::endl;
    std::string dir_script = "mkdir -p " + output_path;
    system(dir_script.c_str());

    //instantiate detectors, matchers
    SURFDetector surf;
    SURFMatcher<BFMatcher> matcher;
    //SURFMatcher<FlannBasedMatcher> matcher; //see what happens
    std::vector<KeyPoint> actual_frame_keypoints, past_frame_keypoints, last_keyframe_keypoints;
    Mat actual_frame_descriptors, past_frame_descriptors, last_keyframe_descriptors;
    std::vector<DMatch> matches;

    //algorithm internals
    std::vector<int> matches_count_list;
    double average_matches_value = -1.0f;
    int average_interval_lenght = cmd.get<int>("t");
    int frame_step = cmd.get<int>("e");
    double matches_distance_threshold = cmd.get<double>("d");
    double sensitivity_threshold = cmd.get<double>("s");
    double noise_threshold = cmd.get<double>("n");

    //print status
    if(verbose){
        std::cout << "Operating with:--------------------------------------------" << std::endl;
        std::cout << "sensitivity_threshold:" << sensitivity_threshold << std::endl;
        std::cout << "noise_threshold:" << noise_threshold << std::endl;
        std::cout << "matches_distance_threshold:" << matches_distance_threshold << std::endl;
        std::cout << "average_interval_lenght video:" << average_interval_lenght << std::endl;
        std::cout << "frame step:" << frame_step << std::endl;
        std::cout << "-----------------------------------------------------------" << std::endl;
    }

    //state variables
    Mat actual_frame;
    int frame_counter = 0;
    int sensitivity_good_matches = -1;
    int noise_good_matches = -1;
    bool is_running = true;
    bool is_keyframe = false;
    char save_frame_name[2000];
    
    //handle display window
    if(verbose) std::cout << "press space to save a picture. q or esc to quit" << std::endl;
    
    std::string window_name = "video | q or esc to quit";
    if(display_video){
        namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
    }
    std::cout << "Processing ..." << std::endl;

    bool is_first_frame = true;
    while(is_running )
    {
        capture >> actual_frame;
        //resize(original_frame, actual_frame, Size(320, 240), 0, 0, INTER_CUBIC);
        if (actual_frame.empty()){
            is_running = false;
            std::cout << "No more frames to process" << std::endl;
            break;
        }

        if(is_first_frame)
        {
            //first frame
            surf(actual_frame, Mat(), past_frame_keypoints, past_frame_descriptors);
            last_keyframe_descriptors = past_frame_descriptors;
            last_keyframe_keypoints = past_frame_keypoints;
            is_keyframe = true;
            is_first_frame = false;
        }
        else
        {
            //all frames but first

            //check step
            if(0 == (frame_counter % frame_step)){
                actual_frame_keypoints.clear();
                surf(actual_frame, Mat(), actual_frame_keypoints, actual_frame_descriptors);
                if(verbose)
                    std::cout <<"-------Frame: " << frame_counter << " - found: " << actual_frame_keypoints.size() << " keypoints on actual frame" << std::endl;

                matches.clear();
                bool potential_keyframe = false;

                //chek that there are descriptors to match
                if((actual_frame_keypoints.size() > 0) && (past_frame_keypoints.size()>0)){
                    matcher.match(actual_frame_descriptors, past_frame_descriptors, matches);

                    sensitivity_good_matches = count_good_matches( matches, matches_distance_threshold);
                    if(verbose)
                        std::cout << "sensitivity_good_matches " <<  sensitivity_good_matches << std::endl;

                    //insert count value into vector
                    matches_count_list.push_back(sensitivity_good_matches);

                    //check vector lenght
                    if(matches_count_list.size() > average_interval_lenght)
                        matches_count_list.erase(matches_count_list.begin());
                    if(verbose)
                        std::cout << "vector size: " << matches_count_list.size() << std::endl;

                    //calculate average
                    int summ = 0;
                    for (unsigned int i = 0; i < matches_count_list.size(); i++) summ = summ + matches_count_list[i];
                    average_matches_value = (double)(summ) / (double)(matches_count_list.size());
                    if(verbose)
                        std::cout << "average_matches_count: " << average_matches_value << std::endl;

                    //avoid division by zero
                    if(0 == average_matches_value) average_matches_value = 1;

                    //calculate sensitivity value
                    double diff_sensitivity = fabs((double)average_matches_value - (double)sensitivity_good_matches) / (double)average_matches_value;
                    if(verbose)
                        std::cout << "actual diff: " << diff_sensitivity << std::endl;

                    if(diff_sensitivity > sensitivity_threshold){
                        if(verbose)
                            std::cout << "potential keyframe" << std::endl;
                        potential_keyframe = true;
                    }
                }

                if(potential_keyframe)
                {
                    //match actual frame and last keyframe
                    matches.clear();

                    //if last keyframe does not have keypoints
                    if(last_keyframe_keypoints.size() == 0){
                        is_keyframe = true;
                    }else{
                        //chek that there are descriptors to match
                        if((actual_frame_keypoints.size() > 0) && (last_keyframe_keypoints.size()>0)){
                            matcher.match(actual_frame_descriptors, last_keyframe_descriptors, matches);
                            noise_good_matches = count_good_matches( matches, matches_distance_threshold);
                            if(verbose)
                                std::cout << "noise_good_matches " <<  noise_good_matches << std::endl;

                            //avoid division by zero
                            if(0 == sensitivity_good_matches) sensitivity_good_matches = 1;

                            //calculate noise value
                            double diff_noise = fabs((double)sensitivity_good_matches - (double)noise_good_matches) / (double)sensitivity_good_matches;
                            if(verbose)
                                std::cout << "diff_noise: " << diff_noise << std::endl;
                            if(diff_noise > noise_threshold){
                                if(verbose)
                                    std::cout << "#### keyframe !!" << std::endl;
                                is_keyframe = true;

                            }
                        }
                    }

                    if(is_keyframe){
                        last_keyframe_descriptors = actual_frame_descriptors;
                        last_keyframe_keypoints = actual_frame_keypoints;
                    }
                }

                //update past frame parameters
                past_frame_keypoints = actual_frame_keypoints;
                past_frame_descriptors = actual_frame_descriptors;
            }
        }

        //wait until averate matches vector is filled
        if((frame_counter > 0)&&(frame_counter < average_interval_lenght)) is_keyframe = false;

        //save keyframe
        if(is_keyframe){
            //std::string save_frame_name = output_path + "/key-frame-" + frame_counter + ".jpg";
            sprintf(save_frame_name, "%s/key-frame-%.3d.jpg",output_path.c_str(), frame_counter);
            imwrite(save_frame_name,actual_frame);
            if(verbose)
                std::cout << "Saved " << save_frame_name << std::endl;

            is_keyframe = false;
        }

        //show image
        if(display_video){
            imshow(window_name, actual_frame);

            //handle input key
            char key = (char) waitKey(20); //delay N millis, usually long enough to display and capture input

            switch (key) {
            case 'q':
            case 'Q':
            case 27: //escape key
                is_running = false;
                break;
            default:
                break;
            }
        }

        //update frame counter
        frame_counter++;
    }

    std::cout << "Done." << std::endl;

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    int result = EXIT_FAILURE;

    //result = demo_matcher(argc, argv);
    result = run_flash_summ(argc, argv);

    return result;
}

