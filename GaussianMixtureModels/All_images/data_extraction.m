

for k=0:2:44
    I=imread(sprintf('pic%d.jpg', k));
    imshow(I);
    mask=roipoly(I);
    green=bsxfun(@times, I, cast(mask, 'like', I));
    imshow(green)
    imwrite(green,sprintf('Green_new/green%d.jpg', k))
end
