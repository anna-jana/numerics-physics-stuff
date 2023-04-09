import numpy as np

bits_per_pixel = 32
bitmap_header_size = 2 + 3 + 2 + 2 + 4
dib_header_size = 40

def get_row_size(image):
    image_width = image.shape[1]
    return 4 * ((bits_per_pixel * image_width + 31) // 32) # in bytes

def get_pixel_array_size(image):
    return image.shape[0] * get_row_size(image)

def get_row_padding(image):
    return get_row_size(image) - image.shape[1] * 4

def write_little_endian(x, out, num_bytes=4):
    out.write(x.to_bytes(4, byteorder="little", signed=False))
    #for i in range(num_bytes):
    #    out.write(bytes([(x >> (i * 8)) & 0xff]))

def write_bitmap_header(image, out):
    # header field fixed values
    out.write(b"\x42")
    out.write(b"\x4d")
    # total size of the image
    total_size = bitmap_header_size + dib_header_size + get_pixel_array_size(image)
    write_little_endian(total_size, out)
    # reserved
    out.write(b"\0" * 4)
    # offset to pixel data
    write_little_endian(dib_header_size, out)

def write_dib_header(image, out):
    # size of dib header
    write_little_endian(dib_header_size, out)
    # dimension of the image
    # shape of the image
    write_little_endian(image.shape[1], out)
    write_little_endian(image.shape[0], out)
    # number of color planes (must be 1)
    out.write(b"\1\0")
    # bits per pixle
    write_little_endian(bits_per_pixel, out)
    # complesssion method 0 = none
    write_little_endian(0, out)
    # size of the image
    write_little_endian(get_pixel_array_size(image), out)
    # pixle per meter
    write_little_endian(1000, out)
    write_little_endian(1000, out)
    # colors in color palette (0 means 2^n)
    write_little_endian(0, out)
    # all colors are important
    write_little_endian(0, out)

def write_pixels(image, out):
    padding = get_row_padding(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # a pixel
            for k in range(4):
                out.write(image[i, j, k])
        # row padding
        out.write(b"0" * padding)

def save_image(filename, image):
    with open(filename, "bw") as out:
        write_bitmap_header(image, out)
        write_dib_header(image, out)
        write_pixels(image, out)

image = np.zeros((200, 100, 4), dtype="byte")
# coordinates start in the lower left corner
image[:, :, 0] = 255 # make image red
image[50:100, 20:50, :] = (255, 0, 255, 0) # add violet rectangle
save_image("test.bmp", image)

